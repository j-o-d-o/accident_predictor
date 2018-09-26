# DLPipe

A machine learning pipline handling data and training for keras models.

## Setup
#### MongoDB
Currently only MongoDB is supported as datastorage for data.

### Workflow
- Clone or Fork the DLPipe Repository
- Add your project to the project folder, there are already a few examples there and a boilerplate which you can copy
- Experiment on your local machine to have a running model
- Sync everything with GIT
- Get a server running to host the centralized DLPipe managment server
- Use the scripts in the scripts folder to deploy the project to the DLPipe managment server
- The server automatically creates a AWS instance which fetches the spcified GIT repo and executes the project
- DLPipe also has a server implemented which communicates with the centralized managment server
- The centralized managment server can be used to get a view of what is happening and all the current jobs (maybe a graphical user interface for uploading jobs there?)
- when job is done, the aws instance is closed and the model is saved in the centralized managment server's mongoDB ready to be downloaded