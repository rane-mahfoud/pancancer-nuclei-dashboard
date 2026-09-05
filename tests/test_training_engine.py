import torch
from torch import nn
from torch.utils.data import DataLoader

from pancancer_nuclei.training.engine import (
    evaluate_semantic_model,
    segmentation_collate,
    train_one_epoch,
)


def make_sample(class_number: int) -> dict[str, torch.Tensor]:
    image = torch.zeros(3, 8, 8)
    image[class_number] = 10.0

    return {
        "image": image,
        "semantic_mask": torch.full(
            (8, 8),
            class_number,
            dtype=torch.long,
        ),
    }


def test_segmentation_collate_builds_batch() -> None:
    batch = segmentation_collate([make_sample(0), make_sample(1)])

    assert batch["image"].shape == (2, 3, 8, 8)
    assert batch["semantic_mask"].shape == (2, 8, 8)


def test_train_one_epoch_changes_model_parameters() -> None:
    samples = [make_sample(1) for _ in range(4)]
    loader = DataLoader(
        samples,
        batch_size=2,
        collate_fn=segmentation_collate,
    )

    model = nn.Conv2d(3, 3, kernel_size=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    original_weights = model.weight.detach().clone()

    loss = train_one_epoch(
        model=model,
        data_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert loss > 0
    assert not torch.equal(model.weight, original_weights)


def test_evaluation_reports_perfect_predictions() -> None:
    samples = [
        make_sample(0),
        make_sample(1),
        make_sample(2),
    ]
    loader = DataLoader(
        samples,
        batch_size=2,
        collate_fn=segmentation_collate,
    )

    metrics = evaluate_semantic_model(
        model=nn.Identity(),
        data_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        number_of_classes=3,
        class_names=("background", "class_a", "class_b"),
    )

    assert metrics["pixel_accuracy"] == 1.0
    assert metrics["macro_foreground_dice"] == 1.0
    assert metrics["dice_per_class"]["class_a"] == 1.0
    assert metrics["dice_per_class"]["class_b"] == 1.0
