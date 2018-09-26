#!/bin/bash

##########
## HELP ##
##########
if [[ ( "$1" == "-h" ) || ( "$1" == "--help" ) ]]; then
    echo "Usage: `basename $0` [-h]"
    echo "  Start DLPipe"
    echo
    echo "  -h, --help      Show this help text"
    echo "  -u, --update    Boolean value [true, false] to force update (set to false if offline), default is true"
    exit 0
fi

###################################
## VARIABLE SETTINGS && DEFAULTS ##
###################################
# force update of packages (takes a little longer)
UPDATE_MODE=true
# specifiy pycharm path to open pycharm with the current conda env
PYCHARM_PATH="/opt/pycharm-community-2018.1/bin/pycharm.sh"
# environment name
ENV_NAME="accident_predictor_env"

#######################
## PARAMETER PARSING ##
#######################
while :
do
    case "$1" in
        -u | --update)
            if [ $# -ne 0 ]; then
                UPDATE_MODE="$2"
            fi
            shift 2
            ;;
        "")
            break
            ;;
        *)
            echo -e "\033[33mWARNING: Argument $1 is unkown\033[0m"
            shift 2 
    esac
done


###########
## START ##
###########
# Create env, update and start pycharm
if [ "$UPDATE_MODE" = true ]; then
    # installing tensorflow as conda can not find it from the environment.yml file
    echo INSTALL TENSORFLOW FROM PIP
    pip install tensorflow

    if [[ $PATH != $ENV_NAME ]]; then
      # Check if the environment exists
      source activate $ENV_NAME
      if [ $? -eq 0 ]; then
        echo ------------ Update Env    ------------
        conda env update -f environment.yml
      else
        # Create the environment and activate
        echo ------------ Create Env  ------------
        conda env create -f environment.yml
        source activate $ENV_NAME
      fi
    fi
else
    source activate $ENV_NAME
fi

KERAS_BACKEND=tensorflow

echo ------------ Start PyCharm ------------
eval $PYCHARM_PATH 2> /dev/null &
