import argparse
from agents.groupAgent import groupAgent
from utils.config import get_config_from_json, setup_logging
import sys, os
from shutil import copyfile

arg_parser = argparse.ArgumentParser(description="")
arg_parser.add_argument('args',
                        metavar='args_json_file',
                        default='None',
                        help='The arguments file in json format')


def main(args_obj):

    # parse the config json file
    args, _ = get_config_from_json(args_obj.args)

    if not os.path.exists(os.path.join(args.exp_log_path, args.exp_name)):
        os.makedirs(os.path.join(args.exp_log_path, args.exp_name, 'logs'))
        os.makedirs(os.path.join(args.exp_log_path, args.exp_name, 'model'))
        os.makedirs(os.path.join(args.exp_log_path, args.exp_name, 'tensorboard'))
        os.makedirs(os.path.join(args.exp_log_path, args.exp_name, 'output'))


    args.log_dir = os.path.join(args.exp_log_path, args.exp_name, 'logs')
    args.model_dir = os.path.join(args.exp_log_path, args.exp_name, 'model')
    args.tensorboard_dir = os.path.join(args.exp_log_path, args.exp_name, 'tensorboard')
    args.output_dir = os.path.join(args.exp_log_path, args.exp_name, 'output')

    setup_logging(args.log_dir, args.mode)

    #Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[args.agent]
    agent = agent_class(args)
    agent.run()
    agent.finalize()


if __name__ == '__main__':

    sys.argv = ['main.py', 'args.json']
    args_obj = arg_parser.parse_args()

    main(args_obj)
