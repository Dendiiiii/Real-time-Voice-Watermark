import os

import soundfile
import yaml
from rich.progress import track
from torch.utils.data import DataLoader

from dataset.data import collate_fn
from dataset.data import wav_dataset as my_dataset
from Distortions.distortions import *
from models import *

seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging_mark = "#" * 20
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main(configs):
    logging.info("main function")
    process_config, model_config, train_config = configs

    # ------------------- get train dataset
    dev_audios = my_dataset(
        process_config=process_config, train_config=train_config, flag="dev"
    )

    batch_size = 1

    dev_audios_loader = DataLoader(
        dev_audios, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # ---------------- build model
    encoder = SimpleEncoder()
    decoder = SimpleDecoder()
    detector = SimpleDetector()

    wm_generator = WatermarkModel(
        encoder=encoder, decoder=decoder, model_config=model_config
    ).to(device)
    wm_detector = WatermarkDetector(
        detector=detector, nbits=0, model_config=model_config
    ).to(device)

    path_model = model_config["test"]["model_path"]
    model_name = model_config["test"]["model_name"]
    if model_name:
        model = torch.load(os.path.join(path_model, model_name))
    else:
        index = model_config["test"]["index"]
        model_list = os.listdir(path_model)
        model_list = sorted(
            model_list, key=lambda x: os.path.getmtime(os.path.join(path_model, x))
        )
        model_path = os.path.join(path_model, model_list[index])
        logging.info(model_path)
        model = torch.load(model_path)
        logging.info("model <<{}>> loadded".format(model_path))

    wm_generator.load_state_dict(model["generator"])
    wm_detector.load_state_dict(model["detector"])
    wm_generator.eval()
    wm_detector.eval()

    # ---------------- Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)

    # ---------------- Embedding
    logging.info(logging_mark + "\t" + "Begin Embedding" + "\t" + logging_mark)
    wm_path = train_config["path"]["wm_speech"]
    global_step = 0
    with torch.no_grad():
        for sample in track(dev_audios_loader):
            if global_step > 10:
                break
            global_step += 1
            wav_matrix = sample["matrix"].to(device)
            sample_rate = sample["sample_rate"]
            watermarked_audio, wm = wm_generator(wav_matrix)
            name = sample["name"][0]
            soundfile.write(
                os.path.join(wm_path, name),
                watermarked_audio.cpu().squeeze(0).detach().numpy(),
                samplerate=sample_rate[0],
                format="WAV",
            )
            prob, msg = wm_detector.detect_watermark(watermarked_audio)
            logging.info("-" * 100)
            logging.info(
                "The watermark probability of the Audio file {0} is {1}, and its decoded message is {2}".format(
                    name, prob, msg
                )
            )


if __name__ == "__main__":
    process_config = yaml.load(
        open(r"./config/process.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(r"./config/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(r"./config/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)
    main(configs)
    torch.cuda.empty_cache()
