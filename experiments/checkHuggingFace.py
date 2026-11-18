
from datasets import load_dataset
from pprint import pprint

dataset = load_dataset(
    "Kaphathy/OphGen",
    "DRIVE",
    split="test",
    streaming=True
)

sample = next(iter(dataset))

print("\n===== IMAGE INFORMATION =====")
print("Image ID:", sample["image_id"])
print("Split:", sample["split"])
print("Dimensions:", sample["width"], sample["height"])
print("Source paths:")
pprint(sample["source_paths"])

print("\n===== LABELS =====")
pprint(sample["labels"])

print("\n===== MASKS =====")
for index, mask in enumerate(sample["masks"]):
    print(f"\nMask {index + 1}:")
    pprint(mask)
