from collections import Counter
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_class_audio(MFCCs, model):
    '''
    Predict class based on MFCC samples
    :param MFCCs: Numpy array of MFCCs
    :param model: Trained model
    :return: Predicted class of MFCC segment group
    '''
    MFCCs = MFCCs.reshape(MFCCs.shape[0],1,MFCCs.shape[1],MFCCs.shape[2])
    MFCCs = torch.from_numpy(MFCCs).type(torch.FloatTensor)
    MFCCs = MFCCs.to(device)
    out = model(MFCCs)
    _,y_predicted = out.max(1)
    #y_predicted = model.predict_classes(MFCCs,verbose=0)
    return(Counter(list(y_predicted)).most_common(1)[0][0])


def predict_prob_class_audio(MFCCs, model):
    '''
    Predict class based on MFCC samples' probabilities
    :param MFCCs: Numpy array of MFCCs
    :param model: Trained model
    :return: Predicted class of MFCC segment group
    '''
    MFCCs = MFCCs.reshape(MFCCs.shape[0],MFCCs.shape[1],MFCCs.shape[2],1)
    y_predicted = model.predict_proba(MFCCs,verbose=0)
    return(np.argmax(np.sum(y_predicted,axis=0)))

def predict_class_all(X_train, model):
    '''
    :param X_train: List of segmented mfccs
    :param model: trained model
    :return: list of predictions
    '''
    predictions = []
    for mfcc in X_train:
        predictions.append(predict_class_audio(mfcc, model))
        # predictions.append(predict_prob_class_audio(mfcc, model))
    return predictions

def confusion_matrix(y_predicted,y_test):
    '''
    Create confusion matrix
    :param y_predicted: list of predictions
    :param y_test: numpy array of shape (len(y_test), number of classes). 1.'s at index of actual, otherwise 0.
    :return: numpy array. confusion matrix
    '''
    y_pred = []
    for i in range(len(y_test)):
        y_pred.append(y_predicted[i].cpu().item())
    # print(y_test, y_predicted, y_pred)
    acc = 0
    for i in range(len(y_test)):
        if(y_test[i]==y_pred[i]):
            acc +=1
    print("Accuracy : ",acc,len(y_test))
    y_test , y_predicted = y_test, y_pred
    confusion_matrix = np.zeros((len(y_test),len(y_test)),dtype=int )
    for index, predicted in enumerate(y_pred):
        confusion_matrix[np.argmax(y_test[index])][predicted] += 1
    return(confusion_matrix)

def get_accuracy(y_predicted,y_test):
    '''
    Get accuracy
    :param y_predicted: numpy array of predictions
    :param y_test: numpy array of actual
    :return: accuracy
    '''
    c_matrix = confusion_matrix(y_predicted,y_test)
    return( np.sum(c_matrix.diagonal()) / float(np.sum(c_matrix)))

if __name__ == '__main__':
    pass


