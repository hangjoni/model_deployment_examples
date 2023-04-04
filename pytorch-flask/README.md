## Deploy model as an API via Flask

### To spin up the app

`FLASK_ENV=development FLASK_APP=app.py flask run`

Send POST request to localhost:5000/predict passing image in file field
App will return the class name for the provided image
