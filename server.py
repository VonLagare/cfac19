import flask
import numpy
import pickle
import pygad

# Use pickle to load in the pre-trained model.
fn = 'models/GA.pkl'

model_instance = pickle.load(open(fn,'rb'))

app = flask.Flask(__name__, template_folder='pages')

@app.route('/', methods=['GET', 'POST'])
def main():
    
    if flask.request.method == 'GET':
        return flask.render_template('home.html')
    
    if flask.request.method == 'POST':
        cough = flask.request.form['cough']
        muscleAche = flask.request.form['muscleAche']
        tiredness = flask.request.form['tiredness']
        soreThroat = flask.request.form['soreThroat']
        runnyNose = flask.request.form['runnyNose']
        stuffyNose = flask.request.form['stuffyNose']
        fever = flask.request.form['fever']
        nausea = flask.request.form['nausea']
        vomiting = flask.request.form['vomiting']
        diarrhea = flask.request.form['diarrhea']
        shortnessOfBreath = flask.request.form['shortnessOfBreath']
        muscdifficultyInBreathingleAche = flask.request.form['difficultyInBreathing']
        lostOfTaste = flask.request.form['lostOfTaste']
        lossOfSmell = flask.request.form['lossOfSmell']
        itchyNose = flask.request.form['itchyNose']
        itchyEyes = flask.request.form['itchyEyes']
        itchyMouth = flask.request.form['itchyMouth']
        itchyInnerEar = flask.request.form['itchyInnerEar']
        sneezing = flask.request.form['sneezing']
        pinkEye = flask.request.form['pinkEye']
        
        input_variables = numpy.array([[cough, muscleAche, tiredness, soreThroat, runnyNose, stuffyNose,
                                        fever, nausea, vomiting, diarrhea, shortnessOfBreath, muscdifficultyInBreathingleAche,
                                        lostOfTaste, lossOfSmell, itchyNose, itchyEyes, itchyMouth, itchyInnerEar,
                                        sneezing, pinkEye]])

        array_inputs =  input_variables.astype(numpy.float)
        
        prediction = pygad.nn.predict(last_layer=model_instance, data_inputs=array_inputs)[0]
        
        pred = ''
        if int(prediction) == 0:
            pred = 'Allergy'
        elif int(prediction) == 1:
            pred = 'Cold'
        elif int(prediction) == 2:
            pred = 'Covid-19'
        else:
            pred = 'Flu'
            
        print(pred)
        
        return flask.render_template('result.html', result=str(pred))
    
@app.route('/model', methods=['GET'])
def model():
    if flask.request.method == 'GET':
        return flask.render_template('model.html')
    
@app.route('/source', methods=['GET'])
def source():
    if flask.request.method == 'GET':
        return flask.render_template('source.html')

    
@app.route('/about', methods=['GET'])
def about():
    if flask.request.method == 'GET':
        return flask.render_template('about.html')


if __name__ == '__main__':
    app.run()
