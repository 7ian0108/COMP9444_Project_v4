import logging
import os


# 输出log
def setup_logger(log_name, output_dir):
    log_file = os.path.join(output_dir, f'{log_name}.log')

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(f"Logger initialized. Log file: {log_file}")
    return log_file
