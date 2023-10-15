# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

__all__ = [
    'load_heartrate'
]


def creat_heartrate(dataset, as_series=False):
    """ Heart Rate Data from MIT dataset:
    
    Heart rhythms in health and disease display complex dynamics. 
    The clinical data suggest that concepts developed in nonlinear mathematics, such as bifurcations and chaos,will be appropriate to describe some of these complex phenomena. 
    Careful analysis will be needed to establish the presence of deterministic chaos in cardiac rhythms. 
    Data that appear highly periodic such as normal sinus rhythm may in reality be quite variable. 
    In contrast, chaotic-appearing rhythms such as ventricular fibrillation often contain strong periodicities.
    
    References
    ----------
    
    Goldberger AL, Rigney DR. Nonlinear dynamics at the bedside. In: Glass L, Hunter P, McCulloch A, eds. Theory of Heart: Biomechanics, Biophysics, and Nonlinear Dynamics of Cardiac Function. New York: Springer-Verlag, 1991, pp. 583-605.
    
    --------
    
    A sample of heartrate data borrowed from an `MIT database <http://ecg.mit.edu/time-series/>`_. 
    The sample consists of 150 evenly spaced (0.5 seconds) heartrate measurements.
    There are two groups(health and disease ) with measurements of heart beat rate, i.e., 6*300 measurements for each group.
    
    Parameters
    ----------
    as_series : bool, optional (default=False)
        Whether to return a Pandas series. If False, will return a 1d
        numpy array.

    Original_data:
    
    Returns
    -------
    dataset : array-like, shape=(n_samples,)
        The heartrate vector.
        
    Examples
    --------
    >>> from datetime import datetime, timedelta
    >>> import pandas as pd
    >>> import numpy as np
    >>> dataset = pd.read_csv('HeartRate_MIT.csv',sep=',',header='infer')
    >>> creat_heartrate(dataset)
    --------
    
    """
    
    # Define the start and end time
    start_time = datetime(2023, 10, 14, 0, 0, 0)  # Start at midnight
    T=600
    end_time = datetime(2023, 10, 14, 0, 8, 15)  # End at 11:59:59 PM
    
    # Define the time step (1 second in this case)
    time_step = timedelta(seconds=5)
    
    # Initialize an empty list to store the time series
    time_series = []
    
    # Generate the time series
    current_time = start_time
    n=0
    while n< T:
        time_series.append(current_time.strftime("%H:%M:%S"))  # Format and add to the list
        current_time += time_step
        n += 1
    
    '''
    # Print the time series
    for timestamp in time_series:
        print(timestamp)
    '''
    Dataa=pd.DataFrame(time_series,columns=['Time'])
    DataT=pd.DataFrame(time_series,columns=['Time'])
    for col in range(len(dataset.columns)):
        data= dataset[dataset.columns[col]] 
        data= np.array(data).reshape(3,600)
        # print(data)
        # np.asarray(ds1.outputs).reshape(-1).tolist()
        for n in range(3):
            if col ==0:
                colname = ['Health_'+str(n+1)]
            else:
                colname = ['Disease_'+str(n+1)]
            Data=pd.DataFrame(np.array(pd.DataFrame(data).iloc[n,:]),columns=colname)
            Dataa=pd.concat([Dataa, Data], axis=1)
        Dataaa=pd.concat([DataT, Dataa], axis=1)
    if as_series:
        return pd.Series(data)
    data = pd.DataFrame(data = Dataaa.T.drop_duplicates().T )
    data.to_csv('HeartRate_MIT_Test.csv',index=False) 
    return data
