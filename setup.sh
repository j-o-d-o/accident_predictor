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
# Create/Update env
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

KERAS_BACKEND=tensorflow
