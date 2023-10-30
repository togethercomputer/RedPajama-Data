from pathlib import Path
from tqdm import tqdm
import fasttext
import subprocess

from core.document import Document
from core.quality_signals.utils.classifiers import \
    preprocess_quality_classifier
from core.constants import CCNET_LABEL
from utilities.io import Reader


class FastTextTrainer:
    # cc label
    cc_label = CCNET_LABEL

    # output file naming convention
    output_fmt = "{dataset}.model.bin"

    def __init__(
            self, artifacts_dir, ccnet_data, target_data, target_name,
            samples_per_class, lang
    ):

        # write args to class variables
        self._ccnet_data = ccnet_data
        self._target_data = target_data
        self._samples_per_class = samples_per_class
        self._lang = lang
        self._target_label = f"__label__{target_name}"

        # build output directory
        out_dir = Path(artifacts_dir) / "classifiers" / self._lang
        out_dir.mkdir(parents=True, exist_ok=True)
        self._output = out_dir / self.output_fmt.format(dataset=target_name)
        self._train_data = out_dir / f"{target_name}.data.train"

    def run(self, logger):
        log_prefix = f"{self.__class__.__name__}(" \
                     f"lang={self._lang}, ccdata={self._ccnet_data}, " \
                     f"target_data={self._target_data}, " \
                     f"target_label={self._target_label})"

        train_data_fh = open(self._train_data, "w")

        logger.info(f"{log_prefix} Start building fasttext classifier")

        # write target data
        samples_per_slice = self._samples_per_class // len(self._target_data)
        total_target_samples = 0

        for target_data_fp in self._target_data:
            reader = Reader(schema=[("text", str)])
            total_target_samples += self.__write_train_chunk(
                uri="file://" + str(target_data_fp),
                reader=reader,
                writer=train_data_fh,
                max_samples=samples_per_slice,
                target_label=self._target_label
            )

        logger.info(f"{log_prefix} Number of target "
                    f"samples found: {total_target_samples}")

        # write ccnet data
        reader = Reader(schema=[("text", str)])
        ccnet_samples = self.__write_train_chunk(
            uri="file://" + str(self._ccnet_data),
            reader=reader,
            writer=train_data_fh,
            max_samples=total_target_samples,
            target_label=self.cc_label
        )
        train_data_fh.close()
        logger.info(f"{log_prefix} Total ccnet samples: {ccnet_samples}")

        # shuffle train data
        logger.info(f"{log_prefix} Shuffling train data")
        subprocess.run(
            ["shuf", "-o", str(self._train_data), str(self._train_data)]
        )

        # train fasttext classifier
        model = fasttext.train_supervised(
            input=str(self._train_data), verbose=2
        )
        model.save_model(str(self._output))
        logger.info(f"{log_prefix} Saved model to {self._output}")

    @staticmethod
    def __write_train_chunk(
            uri, reader: Reader, writer, max_samples, target_label
    ):
        num_samples = 0

        for record in tqdm(
                reader.read(uri, max_samples=max_samples, return_idx=False),
                total=max_samples
        ):
            doc = Document(record.text, domain=None)
            text = preprocess_quality_classifier(document=doc)
            writer.write(f"{target_label} {text}\n")
            num_samples += 1

        writer.flush()

        return num_samples
