import torch




class Data_PreProcessing():
    pass


class Training():

    def loss_function(n : int, y : torch.Tensor, t : torch.Tensor) -> torch.Tensor :
        """
        RMSLE 
        Root Mean Squared Logarithmic Error

        n : Number of instances
        y : Prediction
        t : Target
        """

        sle = torch.sum(torch.log(1+y)-torch.log(1+t))**2
        rmsle = torch.sqrt(sle*(1/n))

        return rmsle


class Data_PostProcessing():
    pass



