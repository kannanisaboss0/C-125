#------------------------------C-125----------------------------#
#-----------------------------ToAPI.py----------------------------#

'''
Importing modules:
-~Flask,~request,~jsonify :-flask
-(custom) emulate_predicitor :-ImagePredictorAPI
'''

from flask import Flask,request,jsonify
from ImagePredictorAPI import emulate_predictor 

#Iniitiaing the app
app=Flask(__name__)

#Setting the route to "/alphabet", and only allowing POST methods
@app.route("/alphabet",methods=["POST"])

#Defining a function to get the image and predict it
def GetImageandPredict():

    #Recieving the image
    image_req=request.files.get("alphabet")

    #Sending the image to the imported function <--~from ImagePredictorAPI.py 
    predictor_func=emulate_predictor(image_req)

    #Returning a json object containing the prediction
    return jsonify({
        "prediciton":predictor_func
    },200)

#Verifying whether the route is "__main__" or not
##Case-1 ~the route is "__main__"
if __name__=="__main__":

    #Running the app, with an enabled debug console
    app.run(debug=True)   

#-----------------------------ToAPI.py----------------------------#  
#------------------------------C-125----------------------------#  