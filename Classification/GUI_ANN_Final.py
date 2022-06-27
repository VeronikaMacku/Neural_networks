import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
#------------------------------
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import joblib as jl
import csv
#-----------------------------------------
#Project 1: Basic ANN classifier
#Inputs: real inputs, (x1,x2) e <-4;2>x<2,5>
#Outputs: "Inside","Outside"
#Training set: points fitting 0.4444444*(x1+2)**2+2.3668639*(x2-3)**2 < 1 belong "Inside"
#-----------------------------------------
#BORDER FUNCTION
class ANNmodel:
    def __init__(self,ANNmodelFile = None):
        self.ANNmodelFile = ANNmodelFile
        self.retrainedModelFile = 'retrainedModel.joblib'
        #-------------------------------------------------
        #Border, where 0.4444444*(x1+2)^2+2.3668639*(x2-3)^2 == 1 (aka An Elipse)
        self.x1a0 = -2
        self.x1a = (1/0.4444444)**(1/2)    #length of semi-major axis
        self.x2b0 = 3
        self.x2b = (1/2.3668639)**(1/2)    #length of semi-minor axis
        #-------------------------------------------------
        #FOR GENERATING DATA POINTS
        self.x_min,self.x_max = -4,2
        self.y_min,self.y_max = 2,5
        self.x_size = 600
        #-------------------------------------------------
        self.generateData()
        self.splitTrainTest()
        if self.ANNmodelFile == None:
            self.ANNtrain(True)
        else:
            self.ANNtrain(False)
        self.ANNtest()
        #self.plotPredData()

    def isInside(self, x):
        y = np.zeros((len(x),1))
        for i in range(0,len(x)):
            fx = 0.4444444*(x[i,0]+2)**2+2.3668639*(x[i,1]-3)**2
            if fx < 1:
                y[i] = 1
            else:
                y[i] = 0
        return y
    
    def generateData(self):
        x1 = (self.x_max - self.x_min) * np.random.rand(self.x_size,1) + self.x_min
        x2 = (self.y_max - self.y_min) * np.random.rand(self.x_size,1) + self.y_min
        X = np.concatenate((x1,x2), axis=1)
        #Adding labels
        x1x2_labels = np.array(self.isInside(X)) #Inside = 1, Outside = 0
        Xy_trainset = np.concatenate((X,x1x2_labels), axis=1)

        self.Xy_trainset = Xy_trainset

    def splitTrainTest(self):
        X = self.Xy_trainset[:,:2]
        y = np.array(self.Xy_trainset[:,2],dtype=np.int8)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=0.2)  #0.2 means 20 % of data will be used for testing 

    def ANNtrain(self, retrain = False):
        if retrain == True:
            ann = MLPClassifier(activation='logistic', solver='lbfgs',hidden_layer_sizes=(3,3), max_iter = 500, alpha = 0.1)
            ann.fit(self.X_train,self.y_train)
            jl.dump(ann,self.retrainedModelFile)
            print(f"Model saved to {self.retrainedModelFile}!")
        else:
            ann = jl.load(self.ANNmodelFile)

        self.ann = ann

    def ANNtest(self):
        Y_pred = self.ann.predict(self.X_test)
        print("--------------------------------------------------")
        print("Labels: 1=Inside, 0=Outside")
        print("Predictions")
        print(Y_pred)
        print("True values")
        print(self.y_test)
        self.score = self.ann.score(self.X_test,self.y_test)
        print(f"Successfully Classified: {self.score * 100} %.")
        print("--------------------------------------------------")
        ypred_labels = np.where(Y_pred == 1,"Inside","Outside").reshape(len(Y_pred),1)   #Inside = 1, Outside = 0
        
        self.XYpred = np.concatenate((self.X_test,ypred_labels),axis=1)

    def ANNpredict(self, X_pred):
        Y_pred = self.ann.predict(X_pred)
        print("--------------------------------------------------")
        print("Labels: 1=Inside, 0=Outside")
        print("Predictions")
        print(Y_pred)
        print("--------------------------------------------------")
        ypred_labels = np.where(Y_pred == 1,"Inside","Outside").reshape(len(Y_pred),1)   #Inside = 1, Outside = 0
        
        return np.concatenate((X_pred,ypred_labels),axis=1)

    def plotPredData(self):
        plt.title("Basic ANN Classificator (MultiLayer Perceptron)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim((self.x_min, self.x_max))
        plt.ylim((self.y_min,self.y_max))

        borderEllipse = mp.Ellipse((self.x1a0,self.x2b0),2*self.x1a,2*self.x2b,  fill = False, color = "grey", linestyle ="--")
        plt.gca().add_patch(borderEllipse)

        XYpred_Inside = list()
        XYpred_Outside = list()
        for line in self.XYpred:
            if line[2] == "Inside":
                XYpred_Inside.append([float(line[0]), float(line[1])])
            else:
                XYpred_Outside.append([float(line[0]), float(line[1])])

        X_in = np.array(XYpred_Inside)
        X_out = np.array(XYpred_Outside)

        if len(X_in) > 0:
            plt.plot(X_in[:,0],X_in[:,1], label = "Inside", color = "blue", marker = "o", linestyle=" ")
        if len(X_out) > 0:
            plt.plot(X_out[:,0],X_out[:,1], label = "Outside", color = "red", marker = "x", linestyle=" ")
        plt.legend()

        #Plotting decision borders
        mesh_step = 0.02
        xx,yy = np.meshgrid(np.arange(self.x_min, self.x_max, mesh_step),np.arange(self.y_min, self.y_max, mesh_step))
        if hasattr(self.ann, "decision_function"):
            Z = self.ann.decision_function(np.c_[xx.ravel(),yy.ravel()])
        else:
            Z = self.ann.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
        Z = Z.reshape(xx.shape)
        colorMap = plt.cm.RdBu
        plt.contourf(xx,yy,Z,cmap = colorMap, alpha=0.8)

        plt.show()
#-----------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Basic ANN Classificator (MultiLayer Perceptron)')
        self.setMinimumSize(QSize(600,600))
        self.content = QWidget()
        layout = QGridLayout()
        # -------------------------------------------------------
        #ANN MODEL
        self.ann = ANNmodel('ModelBest.joblib')

        layout.addWidget(QLabel('ANN MODEL'),0,0)
        self.retrain_button = QPushButton('Retrain ANN') 
        self.retrain_button.clicked.connect(self.retrainANN)
        layout.addWidget(self.retrain_button,0,1)

        self.ANNtestDataLabel = QLabel(self)
        b =  len(self.ann.y_test)
        a =  self.ann.score * b
        self.ANNtestDataLabel.setText('Successfully classified: %d/%d of test samples.' % (a, b))
        layout.addWidget(self.ANNtestDataLabel,1,0)

        #Predictions
        self.X_to_predict = list()
        layout.addWidget(QLabel('PREDICTIONS'),3,0)
        self.predict_button = QPushButton('Predict') 
        self.predict_button.clicked.connect(self.predictWithANN)
        layout.addWidget(self.predict_button,3,1)

        self.file_input = QLineEdit('Filename.csv')
        layout.addWidget(QLabel('Load data to be classified from a file (.csv):'),4,0)
        layout.addWidget(self.file_input, 4,1)

        layout.addWidget(QLabel('Add point: '),5,0)
        self.x_1_input = QLineEdit()
        self.x_2_input = QLineEdit()
        layout.addWidget(QLabel('x_1'),6,0)
        layout.addWidget(self.x_1_input, 6,1)
        layout.addWidget(QLabel('x_2'),7,0)
        layout.addWidget(self.x_2_input, 7,1)
        self.add_button = QPushButton('Add point') 
        self.add_button.clicked.connect(self.addPoint)
        layout.addWidget(self.add_button,5,1)

        layout.addWidget(QLabel('OR choose points to be classified by clicking on the canvas.'),8,0)
        
        self.setOfFilesLoaded = set()

        #PLOT
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("button_press_event", self.onLeftClickonCanvas)
        self.axes = self.figure.add_subplot(111)
        self.plotBoundary()
        self.content.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        layout.addWidget(self.canvas,10,0,5,2)

        #Notes
        self.clear_button = QPushButton('Clear Plot') 
        self.clear_button.clicked.connect(self.plotClearPoints)
        layout.addWidget(self.clear_button,19,0,1,2)

        self.ANNfileLabel3 = QLabel(self)
        self.ANNfileLabel3.setText('Start ANN model loaded from ModelBest.joblib.')
        layout.addWidget(self.ANNfileLabel3,20,0)

        self.ANNfileLabel = QLabel(self)
        self.ANNfileLabel.setText('Any retrained model saved as retrainedModel.joblib.')
        layout.addWidget(self.ANNfileLabel,21,0)

        self.ANNfileLabel2 = QLabel(self)
        self.ANNfileLabel2.setText('Classified data saved as ANN_outputs.csv.')
        layout.addWidget(self.ANNfileLabel2,22,0)

        self.content.setLayout(layout)
        self.setCentralWidget(self.content)

    def retrainANN(self):
        self.ann.generateData()
        self.ann.splitTrainTest()
        self.ann.ANNtrain(retrain = True)
        self.ann.ANNtest()
        b =  len(self.ann.y_test)
        a =  self.ann.score * b
        self.ANNtestDataLabel.setText('Successfully classified: %d/%d of test samples.' % (a, b))
        self.plotClearPoints()

    def predictWithANN(self):
        if self.file_input.text() != 'Filename.csv':
            if not(self.file_input.text() in self.setOfFilesLoaded):
                self.setOfFilesLoaded.add(self.file_input.text())
                try:
                    X_to_pred = self.loadDataToClassify(self.file_input.text())
                    self.X_to_predict.extend(X_to_pred.tolist())
                except Exception:
                    self.file_input.setText('Filename.csv')

        try:    
            self.XYpred = self.ann.ANNpredict(np.array(self.X_to_predict))
            print(self.XYpred)
            self.saveClassData(self.XYpred)
            self.plotPredict()
        except:
            QMessageBox.critical(self, 'Error', 'No data to predict.')

    def loadDataToClassify(self, userFile = 'ANN_inputs.csv'):
        X_to_classify = list()
        try:
            with open(userFile,'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                for line in csv_reader:
                    X_to_classify.append([float(line[0].replace(',','.')),float(line[1].replace(',','.'))])
        except Exception:
            QMessageBox.critical(self, 'Error', 'The file does not exist or cannot be read.')
        else:
            return np.array(X_to_classify)

    def saveClassData(self, XYpred, output_file = 'ANN_outputs.csv'):
        with open(output_file,'w', newline='') as new_file:
            csv_writer = csv.writer(new_file, delimiter=';')
            for i in range(0,len(XYpred)):                #Save into a file in the format: x1;x2;label
                csv_writer.writerow(XYpred[i])

            print(f"Classified data saved to '{output_file}'.")

    def addPoint(self):
        try:
            if (self.x_1_input.text() != None) and (self.x_2_input.text()!=None):
                x1 = float(self.x_1_input.text())
                x2 = float(self.x_2_input.text())
                self.axes.plot(x1,x2,'gx')
                self.canvas.draw()
                self.X_to_predict.append([x1,x2])
        except ValueError as e:
            QMessageBox.critical(self, 'Error', 'x_1 or x_2 is not a number: \n ValueErrors: '+str(e))

    def onLeftClickonCanvas(self, event):
        x1,x2 = event.xdata, event.ydata
        if x1 != None and x2 != None:
            #print(x1,x2)
            self.axes.plot(x1,x2,'gx')
            self.canvas.draw()
            self.X_to_predict.append([x1,x2])

    def plotPredict(self):
        self.axes.clear()
        self.plotBoundary()

        XYpred_Inside = list()
        XYpred_Outside = list()
        for line in self.XYpred:
            if line[2] == "Inside":
                XYpred_Inside.append([float(line[0]), float(line[1])])
            else:
                XYpred_Outside.append([float(line[0]), float(line[1])])

        X_in = np.array(XYpred_Inside)
        X_out = np.array(XYpred_Outside)

        if len(X_in) > 0:
            self.axes.plot(X_in[:,0],X_in[:,1], label = "Inside", color = "blue", marker = "o", linestyle=" ")
        if len(X_out) > 0:
            self.axes.plot(X_out[:,0],X_out[:,1], label = "Outside", color = "red", marker = "x", linestyle=" ")
        self.axes.legend()

        self.canvas.draw()

    def plotBoundary(self):
        #Axis labels, limits
        self.axes.set_xlabel('x1')
        self.axes.set_ylabel('x2')
        self.axes.set_xlim(self.ann.x_min, self.ann.x_max)
        self.axes.set_ylim(self.ann.y_min,self.ann.y_max)
        #Plotting border function
        borderEllipse = mp.Ellipse((self.ann.x1a0,self.ann.x2b0),2*self.ann.x1a,2*self.ann.x2b,  fill = False, color = "grey", linestyle ="--")
        self.figure.gca().add_patch(borderEllipse)
        #Plotting ANN decision borders
        mesh_step = 0.02
        xx,yy = np.meshgrid(np.arange(self.ann.x_min, self.ann.x_max, mesh_step),np.arange(self.ann.y_min, self.ann.y_max, mesh_step))
        if hasattr(self.ann.ann, "decision_function"):
            Z = self.ann.ann.decision_function(np.c_[xx.ravel(),yy.ravel()])
        else:
            Z = self.ann.ann.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
        Z = Z.reshape(xx.shape)
        colorMap = plt.cm.RdBu
        self.axes.contourf(xx,yy,Z,cmap = colorMap, alpha=0.8)
        self.axes.grid()

    def plotClearPoints(self):
        self.X_to_predict.clear()
        self.file_input.setText('Filename.csv')
        self.setOfFilesLoaded.clear()
        self.axes.clear()
        self.plotBoundary()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()