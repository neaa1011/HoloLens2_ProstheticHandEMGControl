"""
Created on Sun Jun  4 21:41:46 2023

@author: Aaron
"""

def TrainModel(model, X_train, y_train, X_test, y_test, cp_callback, es_callback):
    model.fit(    
        X_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[cp_callback, es_callback]
    )
                    
                    
                    
                
                                   
    
            
    
    

    
    
    