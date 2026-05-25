import os
import argparse
from tqdm import tqdm

VALID_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


def find_image(image_dir, filename):
    path = os.path.join(image_dir, filename)
    if os.path.exists(path):
        return path
    base, _ = os.path.splitext(filename)
    for ext in VALID_EXTS:
        alt = os.path.join(image_dir, base + ext)
        if os.path.exists(alt):
            return alt
    return None


def parse_log(log_file):
    invalid_images = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            image_file = parts[0]
            valid = parts[1].lower() == "true"
            if not valid:
                invalid_images.append(image_file)
    return invalid_images


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(
        description="Export invalid HLoc images to HTML gallery"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="invalid_images.html",
    )
    args = parser.parse_args()

    #
    invalid_images = parse_log(args.log_file)
    print(f"Found {len(invalid_images)} invalid images")
    html = []
    html.append("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Invalid Images</title>

<style>
body {
    font-family: Arial;
    margin: 20px;
    background: #111;
    color: white;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
}

.card {
    background: #222;
    padding: 10px;
    border-radius: 8px;
}

img {
    width: 100%;
    height: auto;
    border-radius: 4px;
}

.filename {
    margin-bottom: 8px;
    word-break: break-all;
    font-size: 14px;
}
</style>
</head>

<body>
<h1>Invalid Images</h1>
<div class="grid">
""")

    #
    for image_name in tqdm(invalid_images, desc="Building HTML"):
        img_path = find_image(args.image_dir, image_name)

        if img_path is None:
            continue

        rel_path = os.path.relpath(img_path, os.path.dirname(args.output))

        html.append(f"""
<div class="card">
    <div class="filename">{image_name}</div>
    <a href="{rel_path}" target="_blank">
        <img loading="lazy" src="{rel_path}">
    </a>
</div>
""")

    html.append("""
</div>
</body>
</html>
""")
    with open(args.output, "w") as f:
        f.write("\n".join(html))
    print(f"Saved HTML: {args.output}")
