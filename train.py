import gc
import os

from train.train import Trainer


def main(**kwargs):
    t = Trainer(**kwargs)
    # TODO remove the mode
    mode = kwargs["mode"]
    if mode == "full":
        t.pretrain_generator()
        gc.collect()
        t.train_gan()
    elif mode == "pretrain":
        t.pretrain_generator()
    elif mode == "gan":
        t.train_gan()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="gan",
                        choices=["full", "pretrain", "gan"])
    parser.add_argument("--dataset_name", type=str, default="animeGAN")
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_size", type=int, default=8)
    parser.add_argument("--source_domain", type=str, default="A")
    parser.add_argument("--target_domain", type=str, default="B")
    parser.add_argument("--gan_type", type=str, default="lsgan", choices=["gan", "lsgan"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--reporting_steps", type=int, default=100)
    parser.add_argument("--content_lambda", type=float, default=.4)
    parser.add_argument("--style_lambda", type=float, default=25.)
    parser.add_argument("--g_adv_lambda", type=float, default=8.)
    parser.add_argument("--d_adv_lambda", type=float, default=1)
    parser.add_argument("--generator_lr", type=float, default=1.5e-5)
    parser.add_argument("--discriminator_lr", type=float, default=1e-5)
    parser.add_argument("--ignore_vgg", action="store_true")
    parser.add_argument("--pretrain_learning_rate", type=float, default=1e-4)
    parser.add_argument("--pretrain_epochs", type=int, default=2)
    parser.add_argument("--pretrain_saving_epochs", type=int, default=1)
    parser.add_argument("--pretrain_reporting_steps", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--checkpoint_dir", type=str, default="training_checkpoints")
    parser.add_argument("--generator_checkpoint_prefix", type=str, default="generator")
    parser.add_argument("--discriminator_checkpoint_prefix", type=str, default="discriminator")
    parser.add_argument("--pretrain_checkpoint_prefix", type=str, default="pretrain_generator")
    parser.add_argument("--pretrain_model_dir", type=str, default="models")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--disable_sampling", action="store_true")
    # TODO: rearrange the order of options
    parser.add_argument(
        "--pretrain_generator_name", type=str, default="pretrain_generator"
    )
    parser.add_argument("--generator_name", type=str, default="generator")
    parser.add_argument("--discriminator_name", type=str, default="discriminator")
    parser.add_argument("--not_show_progress_bar", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show_tf_cpp_log", action="store_true")

    args = parser.parse_args()

    if not args.show_tf_cpp_log:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    args.show_progress = not args.not_show_progress_bar
    kwargs = vars(args)
    main(**kwargs)
