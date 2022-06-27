import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense
#------------------------------------------------------
#APPROXIMATED FUNCTIONS
#Rastrigin, Input domain: (-5.12,5.12), Global minimum f(x*) = 0 at x*=(0,...,0)
def rastrigin_2d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    A = 10
    return A*2 + (X**2 - A*np.cos(2*np.pi*X)) + (Y**2 - A*np.cos(2*np.pi*Y))

#Schwefel, Input domain:(-500,500), Global minimum f(x*) = 0 at x*=(420.9687,...,420.9687)
def schwefel_2d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return 418.9829*2 - ( X*np.sin(np.absolute(X)**(1/2))  + Y*np.sin(np.absolute(Y)**(1/2)))
#------------------------------------------------------
#CREATING THE TRAINING SET+TESTING SET
N_train = 50000
N_test = N_train//5
print(" ")
choice = input("Enter 'r' for Rastrigin function or 's' for Schwefel function: ")
if choice == "r":
    functionName = "Rastrigin"
    x_min = -5.12     #lower bound
    x_max = 5.12      #upper bound

    #Training set
    x_ts = (x_max - x_min) * np.random.rand(N_train,2) + x_min
    F_ts = np.reshape(rastrigin_2d(x_ts[:, 0], x_ts[:, 1]), (x_ts.shape[0], 1))
    #Testing set
    x_test = (x_max - x_min) * np.random.rand(N_test,2) + x_min
    F_test = np.reshape(rastrigin_2d(x_test[:, 0], x_test[:, 1]), (x_test.shape[0], 1))
else:
    functionName = "Schwefel"
    x_min = -500     #lower bound
    x_max = 500      #upper bound

    #Training set
    x_ts = (x_max - x_min) * np.random.rand(N_train,2) + x_min
    F_ts = np.reshape(schwefel_2d(x_ts[:, 0], x_ts[:, 1]), (x_ts.shape[0], 1))
    #Testing set
    x_test = (x_max - x_min) * np.random.rand(N_test,2) + x_min
    F_test = np.reshape(schwefel_2d(x_test[:, 0], x_test[:, 1]), (x_test.shape[0], 1))
#------------------------------------------------------
#MODEL TRAINING
net = Sequential(name=functionName)
net.add(Dense(100, input_dim=2, activation='tanh'))
net.add(Dense(200, activation='tanh'))
net.add(Dense(1, activation='linear'))
if choice == "r":
    optim='sgd'
    net.compile(loss='mse', optimizer=optim, metrics=['mean_squared_error'])
    num_epochs = 100
    b_size = 64
    #In 4 of 5 cases, after x-th iteration, the test acc[mse] was: <25 (1st iter.), <10 (2nd), <4 (3rd)
    #In 1 of 5 cases, the model loss decreased slowly and had a final test acc[mse] of about 70. 
    #Note: If in the first 20 epochs of model training, the loss[mse] is not <100, the last case has occured and it is recommended to restart the training in this case.
else:
    optim='adam'
    net.compile(loss='mse', optimizer=optim, metrics=['mean_squared_error'])
    num_epochs = 300
    b_size = 256
    #In 5 of 5 cases after x-th iteration the test acc[mse] was: <21k (1st iter.), <6k (2nd), <2.5k (3rd)
#------------------------------------------------------
#MODEL ADAPTATIONS
R = 3                   #The entire process (all iterations) took about 6 minutes on the test PC.
for i in range(1,R+1):
    print("-------------------------------")
    print(f"Iteration: {i}/{R}")
    print("-------------------------------")

    #MODEL PARAMETERS    
    print("-------------------------------")
    print("Sequential ANN")
    net.summary()
    print("ANN optimized with: ", optim)
    search_choice = "Grid"
    print("Global minimum searched with:", search_choice,"search")

    #FITTING
    print("-------------------------------")
    net.fit(x_ts, F_ts, epochs=num_epochs, batch_size=b_size)
    print("-------------------------------")
   
    #FINDING GLOBAL OPTIMUM
    print("-------------------------------")
    print(f"Iteration: {i}/{R}")
    print("-------------------------------")
    print("Searching for minimum of the ANN model...")
    if search_choice == "Random":
        #RANDOM SEARCH
        N_search = 200
        x_search = (x_max - x_min) * np.random.rand(N_search,2) + x_min
        x_best = np.array(x_search[0,:],ndmin=2)
        F_best = net.predict(x_best)
        for j in range(0,N_search):
            x_new = np.array(x_search[j,:],ndmin=2)
            F_new = net.predict(x_new)
            if (F_best > F_new):
                x_best = x_new
                F_best = F_new
            if (((j+1) % 50) == 0):
                print(f"Evaluated points {j+1}/{N_search}: F*opt={F_best} at x*opt={x_best}.")
    elif search_choice == "Grid":
        #GRID SEARCH
        N_search = 30
        v = np.linspace(x_min, x_max, N_search)
        X1, X2 = np.meshgrid(v, v)

        x_best = np.array([[X1[0,0], X2[0,0]]])
        F_best = net.predict(x_best)
        for k in range(N_search):
            for l in range(N_search):
                x_new = np.array([[X1[k,l], X2[k,l]]])
                F_new = net.predict(x_new)
                if (F_best > F_new):
                    x_best = x_new
                    F_best = F_new
                if (((k*N_search+l+1) % 150) == 0):
                    print(f"Evaluated points {k*N_search+l+1}/{N_search*N_search}: F*opt={F_best} at x*opt={x_best}.")
    print("-------------------------------")

    #COMPARISON
    if choice == 'r':
        F_best_true = np.reshape(rastrigin_2d(x_best[:, 0], x_best[:, 1]), (1, 1))
    else:
        F_best_true = np.reshape(schwefel_2d(x_best[:, 0], x_best[:, 1]), (1, 1))

    #ADD THE POINT TO THE TRAINING SET
    x_ts = np.concatenate((x_ts,x_best), axis = 0)
    F_ts = np.concatenate((F_ts,F_best_true), axis = 0)

    #INFO
    print(f"Global minimum of {functionName}:")
    if choice == "r":
        print("F(x_opt) = 0 at x_opt=[0,0]")
    else:
        print("F(x_opt) = 0 at x_opt=[420.9687,420.9687]")
    print("Approximation of minimum:")
    print(f"F*opt={F_best} at x*opt={x_best}.")
    print(" ")
    print("Difference:")
    print(f"Global minimum: F*opt - F(x_opt) = {F_best - 0}")
    print(f"Local value: F*opt - F(x*opt) = {F_best - F_best_true}")
    print(" ")
    print(f"Overall ANN loss and accuracy after iteration {i}/{R}:")
    results = net.evaluate(x_test, F_test, batch_size=b_size, verbose = 0)
    print("Test loss[mse], Test acc [mse]: ", results)
    print("-------------------------------")
#---------------------------------------------------
#Predictions
N_pred = 30
v = np.linspace(x_min, x_max, N_pred)
X1, X2 = np.meshgrid(v, v)
if choice == 'r':
    Z = rastrigin_2d(X1, X2)
else:
    Z = schwefel_2d(X1, X2)

Z_pred = np.zeros((N_pred, N_pred))
for i in range(N_pred):
    for j in range(N_pred):
        Z_pred[i, j] = net.predict(np.array([[X1[i,j], X2[i,j]]]))
#---------------------------------------------------
#PLOT
fig = plt.figure()
fig.suptitle("Functions")

#Subplot - 1 - Original function
ax = fig.add_subplot(131, projection='3d')
ax.set_title(functionName + " function")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_xticks([x_min,0,x_max])
ax.set_yticks([x_min,0,x_max])
ax.plot_surface(X1, X2, Z)
ax.grid()

#Subplot - 2 - ANN approximated function
ax2 = fig.add_subplot(132, projection='3d')
ax2.set_title("Final " + functionName + " function approximation")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_xticks([x_min,0,x_max])
ax2.set_yticks([x_min,0,x_max])
ax2.plot_surface(X1, X2, Z_pred)
ax2.grid()

#Subplot - 2 - ANN approximated function Error
ax3 = fig.add_subplot(133, projection='3d')
ax3.set_title("Final " + functionName + " function approximation  - Error")
ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
ax3.set_xticks([x_min,0,x_max])
ax3.set_yticks([x_min,0,x_max])
ax3.plot_surface(X1, X2, Z - Z_pred,cmap=plt.get_cmap('coolwarm'))
ax3.grid()

plt.show()