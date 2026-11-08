"""Modern memory-optimized training with Memora framework."""

from dataset import get_risk_dataset
from model import RiskPredictionModel
from memora import Memora


def main():
    print("=== [Approach 2] Training With Memora Optimization Framework ===")

    # 1. Load dataset & model
    train_dataset, val_dataset = get_risk_dataset()
    model = RiskPredictionModel()

    # 2. Let Memora analyze the hardware and auto-tune everything
    trainer = Memora(
        model=model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size="auto",               # Automatically discovers best batch size
        precision="auto",                # Automatically chooses BF16/FP16/FP32
        gradient_accumulation="auto",    # Configures steps to preserve effective batch
        activation_checkpointing="auto", # Conserves VRAM if tight
        device="auto",                   # CUDA on GPU rig, MPS on Mac, CPU on laptop
        report_output_dir="./reports",
    )

    # 3. Preview what Memora decided and why
    print("\n--- Memora Intelligent Optimization Decisions ---")
    recommendations = trainer.optimize()
    for category, reason in recommendations.reasons.items():
        print(f" • {category.upper():<22}: {reason}")

    # 4. Train model with full profiling
    print("\n--- Training With Memora Engine ---")
    report = trainer.fit(epochs=3)

    # 5. Display professional performance summary
    report.print_summary()


if __name__ == "__main__":
    main()

