from pathlib import Path

DATA_DIR = Path("data/extracted")

fake_count = 0
genuine_count = 0

print("Checking dataset...\n")

for file in DATA_DIR.rglob("*.txt"):
    path = str(file).lower()

    if "deceptive_from_mturk" in path:
        fake_count += 1

    elif "truthful_from_web" in path or "truthful_from_tripadvisor" in path:
        genuine_count += 1

print("Dataset Analysis Complete!")
print("--------------------------")

print("Fake Reviews:", fake_count)
print("Genuine Reviews:", genuine_count)
print("Total Reviews:", fake_count + genuine_count)

sample_file = next(DATA_DIR.rglob("*.txt"), None)

if sample_file:
    print("\nSample Review:")
    print(sample_file.read_text(encoding="utf-8", errors="ignore")[:500])