import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score  # type: ignore[import-untyped]


class DistillationLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.3,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hard_loss = self.ce(student_logits, labels)

        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T**2)

        total = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total, {"hard": hard_loss, "soft": soft_loss, "total": total}


class DistillationTrainer:
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        temperature: float = 4.0,
        alpha: float = 0.3,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.grad_clip = grad_clip

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.criterion = DistillationLoss(
            temperature=temperature,
            alpha=alpha,
            class_weights=class_weights.to(device) if class_weights is not None else None,
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.student.train()
        total_loss, total_hard, total_soft = 0.0, 0.0, 0.0
        correct, total = 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        teacher_logits = self.teacher(images)
                else:
                    teacher_logits = self.teacher(images)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    student_logits = self.student(images)
                    loss, ld = self.criterion(student_logits, teacher_logits, labels)
                assert self.scaler is not None
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                student_logits = self.student(images)
                loss, ld = self.criterion(student_logits, teacher_logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            total_hard += ld["hard"].item()
            total_soft += ld["soft"].item()
            correct += (student_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        n = len(loader)
        return {
            "loss": total_loss / n,
            "hard": total_hard / n,
            "soft": total_soft / n,
            "acc": correct / total,
        }

    @torch.no_grad()
    def validate(self, loader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.student.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    t_logits = self.teacher(images)
                    s_logits = self.student(images)
                    loss, _ = self.criterion(s_logits, t_logits, labels)
            else:
                t_logits = self.teacher(images)
                s_logits = self.student(images)
                loss, _ = self.criterion(s_logits, t_logits, labels)

            total_loss += loss.item()
            preds = s_logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        f1 = f1_score(all_labels, all_preds, average="macro")
        return {"loss": total_loss / len(loader), "acc": correct / total, "f1": f1}
