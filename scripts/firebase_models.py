"""Firebase Model Link Manager — populate Firestore with fast download mirrors."""

from __future__ import annotations
import sys
from pathlib import Path

import yaml


def populate_firebase(cred_path: str, collection: str = "comfyforge_models"):
    """Upload all model entries from models.yaml to Firestore for fast lookup."""
    import firebase_admin
    from firebase_admin import credentials, firestore

    cred = credentials.Certificate(cred_path)
    app = firebase_admin.initialize_app(cred)
    db = firestore.client()

    registry = yaml.safe_load(Path("configs/models.yaml").read_text())
    models = registry.get("models", [])

    batch = db.batch()
    for m in models:
        doc_ref = db.collection(collection).document(m["id"])
        batch.set(doc_ref, {
            "id": m["id"],
            "filename": m["filename"],
            "type": m["type"],
            "size_gb": m.get("size_gb", 0),
            "tags": m.get("tags", []),
            "download_url": m["sources"][0] if m.get("sources") else "",
            "mirror_url": "",  # Fill with your own fast mirrors
        }, merge=True)
        print(f"  → {m['id']}")

    batch.commit()
    print(f"\n✓ Uploaded {len(models)} models to Firestore/{collection}")


def set_mirror(cred_path: str, model_id: str, mirror_url: str, collection: str = "comfyforge_models"):
    """Set a custom mirror URL for a model (e.g., Firebase Storage, GCS, S3)."""
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    db.collection(collection).document(model_id).update({
        "download_url": mirror_url,
    })
    print(f"✓ Set mirror for {model_id} → {mirror_url}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python firebase_models.py populate <cred.json>")
        print("  python firebase_models.py mirror <cred.json> <model_id> <url>")
        sys.exit(1)

    cmd = sys.argv[1]
    cred = sys.argv[2]

    if cmd == "populate":
        populate_firebase(cred)
    elif cmd == "mirror":
        set_mirror(cred, sys.argv[3], sys.argv[4])
