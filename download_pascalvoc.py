import os
import tarfile
import torchvision.datasets as datasets


def setup_manual_pascal():
    root_dir = "./data"
    tar_path = os.path.join(root_dir, "VOCtrainval_11-May-2012.tar")

    # 1. Verifica esistenza file
    if not os.path.exists(tar_path):
        print(f"ERRORE: Non trovo il file in {tar_path}")
        print(
            "Per favore scaricalo manualmente da: http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"
        )
        print("E mettilo nella cartella 'data'.")
        return

    # 2. Verifica dimensione (deve essere ~2GB)
    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"File trovato. Dimensione: {size_mb:.2f} MB")

    if size_mb < 1000:
        print("ERRORE: Il file è troppo piccolo! È sicuramente corrotto.")
        print("Cancellalo e riscaricalo dal browser.")
        return

    # 3. Estrazione Manuale
    extract_path = os.path.join(root_dir, "VOCdevkit")
    if not os.path.exists(extract_path):
        print("Estrazione dell'archivio in corso (può richiedere qualche minuto)...")
        try:
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=root_dir)
            print("Estrazione completata!")
        except Exception as e:
            print(f"Errore nell'estrazione: {e}")
            return
    else:
        print("Cartella 'VOCdevkit' già presente. Salto l'estrazione.")

    # 4. Test Caricamento
    print("Test caricamento dataset...")
    try:
        # download=False è cruciale qui: dice a torchvision "fidati, i file sono lì"
        dataset = datasets.VOCSegmentation(
            root=root_dir, year="2012", image_set="val", download=False
        )
        print(f"SUCCESSO! Dataset caricato. Immagini trovate: {len(dataset)}")
    except Exception as e:
        print(f"Errore Torchvision: {e}")
        print("Assicurati che la struttura sia: data/VOCdevkit/VOC2012/...")


if __name__ == "__main__":
    setup_manual_pascal()
