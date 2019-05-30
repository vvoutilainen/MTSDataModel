import pandas as pd
import numpy as np

class MyException(Exception):
    pass

class MTSDataModel:
    """
    Class for data storing and maniplations data in multi-variate
    time series format.
    """
    def __init__(self, filepath, colnames):
        """
        Read in data in csv format from path filepath.
        Input colnames is a dict that dictates column names.
        """
        # Read in data 
        self.datain = pd.read_csv(filepath)
        self.colnames = colnames
        self.df = self.datain.pivot_table(values = colnames['value'], index = colnames['index'], columns = [colnames['level1'],colnames['level2']]).copy()
        self.df.columns.names = [None,None]
        self.df.index.name = None

    ##############################################################
    # Static functions
    ##############################################################

    @staticmethod
    def ConstructWvltFunctions():
        """
        R function(s) for wavelet MRA decomposition. Requires package wavelets.
         - WaveletMRA(): Performs MODTW MRA decomposition of variable in frame X.

        Returns fuctions in a rpy2 type package. 
        """
        from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
        wvltstring = """
        WaveletMRA = function (X, levels, filter='la8'){
            # R function that returns wavelet MRA series for given series X.
            # X is assumed to be a data frame with single series.
            
            require(wavelets)

            wav = mra(X, filter=filter, n.levels=levels, boundary="reflection", fast=TRUE, method="modwt")
            D = as.data.frame(slot(wav, "D"))
            D = D[1:(nrow(D)/2),]
            S = as.data.frame(slot(wav, "S"))
            S = S[1:(nrow(S)/2),]

            mraseries = cbind(D)
            mraseries[[tail(colnames(S), n=1)]] = S[[tail(colnames(S), n=1)]]
            for (detail in grep('D', colnames(mraseries), value=TRUE)){
                names(mraseries)[names(mraseries) == detail] = paste0('wl',substr(detail, 2, 2))
            }
            for (smooth in grep('S', colnames(mraseries), value=TRUE)){
                names(mraseries)[names(mraseries) == smooth] = paste0('wl',as.character(as.integer(substr(detail, 2, 2)) + 1) )
            }
            
            return(mraseries)
        }
        """
        wvltpackage = SignatureTranslatedAnonymousPackage(wvltstring, "powerpack")        
        return wvltpackage
            
    @staticmethod
    def ExpandingSampleCalc(fun,frame,minobsamount,**kwargs):
        """
        Function designed to perform calculations on "exapnding" sample. This means that,
        given pre-sample, we increment variable one obseravtion at the time and re-calculate
        desired methid with one more observation.

        First function truncates input frame to used sample (depends on minobamount), then
        performs calculation using (funtional) argument fun. Returns two types of results:
            - frame containing last values of each step, that is variable calculated on
              "expanding" sample
            - frame containing variables calculated from the whoel reduced sample. 
        """
        # Get expanding sample dates. 1st date will be such that 1st sample contains
        # minobsamount observations.
        expdates = frame.tail(len(frame)-(minobsamount-1)).index
        # Loop over remaining observation dates. First crtframe in loop will
        # contain minobsamount observations
        counter = 0
        for d in expdates:
            # Get crt frame
            crtframe = frame[frame.index <= d].copy()

            # Apply chosen calculation function to expanding sample
            crtresultframe = fun(crtframe, **kwargs)
            
            # Append suffix "_exp" for expanding variables
            crtresultframe.columns = pd.MultiIndex.from_tuples([(x[0]+"_exp", x[1]) for x in crtresultframe.columns])
            
            if counter==0:
                # Create exapanding resultframe
                resultframeexp  = crtresultframe[crtresultframe.index == d]
                # Append suffix of end date reduced full variables
                crtresultframe.columns = pd.MultiIndex.from_tuples([(x[0]+"_"+d, x[1]) for x in crtresultframe.columns])                
                # Create reduced full resultframe with same index and columns as input frame
                resultframefull = pd.DataFrame(index=frame.index)
                # Append variables to reduced full resultframe. Thos merge throws a warning but seems to work as intended...
                resultframefull = pd.merge(resultframefull, crtresultframe, left_index = True, right_index = True, how = 'left')                
            else:
                # Append last values of crtresultframe to exapanding resultframe
                resultframeexp  = resultframeexp.append(crtresultframe[crtresultframe.index == d])
                # Append suffix of end date reduced full variables
                crtresultframe.columns = pd.MultiIndex.from_tuples([(x[0]+"_"+d, x[1]) for x in crtresultframe.columns])
                # Append variables to reduced full resultframe
                resultframefull = pd.merge(resultframefull, crtresultframe, left_index = True, right_index = True, how = 'left')
            counter = counter + 1
        return resultframeexp, resultframefull

    @staticmethod
    def MRADecompose(frame,entity,var,wvltpckg,levels,filter):
        """Function for MRA decomposition"""
        from rpy2.robjects import pandas2ri
        # Frame with single variable. If crtseires is not in frame, should throw error later
        crtframe = frame.iloc[:, (frame.columns.get_level_values(0).isin([var])) & (frame.columns.get_level_values(1)==entity)]
        # Truncate NAs from single variable, needed for mra function to run as it cannot
        # handle NA values
        crtframe = crtframe.dropna(axis = 0, how = 'any')
        # Apply wavelet mra
        pandas2ri.activate()
        mraseries = wvltpckg.WaveletMRA(crtframe,levels=levels, filter=filter)
        mraseries = pandas2ri.ri2py_dataframe(mraseries)
        mraseries.columns = [var+'_'+x for x in mraseries.columns]
        mraseries.index = crtframe.index
        # Modify to multi-columned
        mraseries.columns = pd.MultiIndex.from_tuples(list(zip(mraseries.columns,[entity]*len(mraseries.columns))))              
        return mraseries

    @staticmethod
    def Deflate(frame):
        '''
        Given input pandas DataFrame with columns
            X: series to be deflated
            I: consumer price index,
        deflates series X using I. Index is asummed to represent dates.

        It is assumed that individual series may contain NAs at start and
        end but not in the middle. Output series can only have dates common
        in X and I. Returns pandas series with deflated series values, with
        same date range as in input frame. Missing values in returned series
        are NAs.
        '''
        redframe = frame.dropna(axis = 0, how = 'any').copy()
        redframe['X_hat'] = redframe['X'] * (redframe.iloc[0,list(redframe.columns).index('I')] /  redframe['I'])
        frame = pd.merge(frame, redframe['X_hat'], left_index = True, right_index = True, how = 'left')    
        return frame['X_hat']        

    ##############################################################
    # Methods
    ##############################################################

    def ReturnDf(self):
        """Return df."""
        return self.df.copy()

    def PlotVariables(self, variables, entities, ax=None):
        """
        Plot given series from frame
        """
        self.df.iloc[:, (self.df.columns.get_level_values(0).isin(variables)) & (self.df.columns.get_level_values(1).isin(entities))].plot(ax=ax)

    def GetVariables(self, variables, entities=None):
        """
        Get given series from frame
        """
        if entities == None:
            return self.df.iloc[:, self.df.columns.get_level_values(0).isin(variables)]
        else:
            return self.df.iloc[:, (self.df.columns.get_level_values(0).isin(variables)) & (self.df.columns.get_level_values(1).isin(entities))]

    def DropVariables(self, variables, entities=None):
        """
        Drop variables from frame in-place.
        """
        if entities == None: 
            self.df.drop(variables, axis=1, level=0, inplace=True)
        else:
            from itertools import product as iterprod
            self.df.drop(list(iterprod(variables, entities)), axis=1, inplace=True)        

    def DeflateVariables(self, variables, entities=None, infvar = 'Inflation'):
        """
        For selected 1st level variables and 2nd level entities, deflate chosen
        variables with chosen inflation variable. 
        
        Defaults to using all 2nd level entities and variable "Inflation" as
        inflation variable.

        Adds deflated series into df with suffix '_def'.
        """
        for seriestodefl in variables:
            if entities == None:
                entities = list(np.unique(self.df.columns.get_level_values(1).values))          
            counter = 0
            for entity in entities:

                # Prepare for deflation
                frame = self.df.iloc[:, (self.df.columns.get_level_values(0).isin([seriestodefl,infvar])) & (self.df.columns.get_level_values(1)==entity)]
                frame.columns = [a for a,b in frame.columns.get_values()]
                frame.rename(columns = {infvar: "I", seriestodefl: "X"}, inplace = True)

                # Deflate
                frame['X_hat'] = self.Deflate(frame)

                # Modify back to multi-columned
                frame.rename(columns = {"I": infvar, "X":seriestodefl, "X_hat":seriestodefl+'_def'}, inplace = True)
                frame.columns = pd.MultiIndex.from_tuples(list(zip(frame.columns,[entity]*len(frame.columns))))

                # Save deflated series
                if counter == 0:
                    final_frame = frame[[seriestodefl+'_def']].copy()
                else:
                    final_frame = pd.merge(final_frame, frame[[seriestodefl+'_def']], left_index = True, right_index = True, how = 'left')                
                counter += 1 
            self.df = pd.merge(self.df, final_frame, left_index = True, right_index = True, how = 'left')
            
    def DetrendVariables(self, variables, difftype = 'ld', entities=None):
        """
        Det-trend chosen variables. De-trending options (difftype) are:
          - ld: 1st log-differences
          - d : 1st difference
        
        Appends de-trended series to data frame with suffix depending on difftype.
        """
        for seriestodt in variables:    
            if entities == None:
                entities = list(np.unique(self.df.columns.get_level_values(1).values))
            # If seriestodt not in frame, should throw error when differencing
            crt_frame = self.df.iloc[:, (self.df.columns.get_level_values(0).isin([seriestodt])) & (self.df.columns.get_level_values(1).isin(entities))]
            if difftype == 'ld':
                crt_frame = np.log(crt_frame).diff()
            elif difftype == 'd':
                crt_frame = crt_frame.diff()
            else:
                raise MyException("Invalid difftype selected.")
            crt_frame.rename(columns={seriestodt:seriestodt+"_"+difftype+"1"}, inplace=True, level = 0)
            self.df = pd.merge(self.df, crt_frame, left_index = True, right_index = True, how = 'left')
        
    def MRADecomposition(self, variables, entities=None, levels=6, filter='la8', minobsamount=40, expanding='none'):
        """
        Wavelet multi-resolution decomposition of given variables, either on
        full or expanding sample. If 
            - expanding == 'none' (default), calculates decomposition on full sample
            - expanding == 'expanding', calculates decomposition on expanding sample and stores expanding series
            - expanding == 'redfull', calculates decomposition on expanding sample and stores reduced full series
            - expanding == 'both', calculates decomposition on expanding sample stores both expanding and reduced full series
            
        In expanding sample calculations uses  minobsamount observations in pre-sample.

        If multiple variables passed in, all variables belonging to
        same entity will be trubcated to same length before estimating 
        wavelet MRA components.

        For each variable/entity/sample end date combination, appends level + 1 new variables to
        data frame corresponding to MODWT MRA details and smooth.
        """
        
        # Get R wavelet functions
        wvltpackage = self.ConstructWvltFunctions()

        # Select entities
        if entities == None:
                entities = list(set([x[1] for x in self.df.columns.get_values()]))

        for entity in entities:
            # Select variables
            if variables == None:
                variables = [x[0] for x in self.df.columns.get_values() if x[1] == entity]
            
            # Select variables under given entity into frame
            frame = self.df.iloc[:, (self.df.columns.get_level_values(0).isin(variables)) & (self.df.columns.get_level_values(1)==entity)]
            
            # Truncate NAs (if any) for all variables under given entity, both from start and end
            frame = frame.dropna(axis = 0, how = 'any')
            
            # Check that we have at least minobsamount observations in full sample. If not, skip entity.
            if len(frame) < minobsamount:
                print("For entity {} number of observations {} less than minobsamount = {}.\nNo MRA decomposition performed!\n-------"
                    .format(entity, len(frame),minobsamount))
                continue
            
            # Loop over variables
            for var in variables:

                # If decomposition on using full sample
                if expanding=='none':
                    # Apply MRA decomposition per each variable
                    resultframe = self.MRADecompose(frame,entity,var,wvltpackage,levels,filter)
                    self.df = pd.merge(self.df, resultframe, left_index = True, right_index = True, how = 'left')
                # If decomposition using expanding sample, storing only expanding
                elif expanding=='expanding':
                    # Apply MRA decomposition to expanding sample per each variable
                    resultframe,_ = self.ExpandingSampleCalc(self.MRADecompose, frame, minobsamount,
                                                              entity=entity, var=var, wvltpckg=wvltpackage, levels=levels,filter=filter)
                    self.df = pd.merge(self.df, resultframe, left_index = True, right_index = True, how = 'left')
                # If decomposition using expanding sample, storing only reduced full samples
                elif expanding=='redfull':
                    # Apply MRA decomposition to expanding sample per each variable
                    _,resultframe = self.ExpandingSampleCalc(self.MRADecompose, frame, minobsamount,
                                                              entity=entity, var=var, wvltpckg=wvltpackage, levels=levels,filter=filter)
                    self.df = pd.merge(self.df, resultframe, left_index = True, right_index = True, how = 'left')

                # If decomposition using expanding sample, storing both expanding and reduced full samples
                elif expanding=='both':
                    # Apply MRA decomposition to expanding sample per each variable
                    resultframeexp,resultframefull = self.ExpandingSampleCalc(self.MRADecompose, frame, minobsamount,
                                                              entity=entity, var=var, wvltpckg=wvltpackage, levels=levels,filter=filter)
                    self.df = pd.merge(self.df, resultframeexp,  left_index = True, right_index = True, how = 'left')
                    self.df = pd.merge(self.df, resultframefull, left_index = True, right_index = True, how = 'left')
                # else throw error
                else:
                    raise MyException("Invalid selection for argument expanding.")
                                
    def SumVariables(self,variables,entities=None):
        """
        Aggregate given variables, under same entity, using simple sum.

        variables is a dict with key being new variable name and value designating
        variables to be summed together.
        """
        for key in variables:

            if entities == None:
                # If no entities selected, get those for which all given variables exists
                entities = []
                all_entities = list(np.unique(self.df.columns.get_level_values(1).values))
                for crtentity in all_entities:
                    if set(variables[key]).issubset(list(self.df.iloc[:, self.df.columns.get_level_values(1).isin([crtentity])].columns.get_level_values(0))):
                        entities.append(crtentity)

            for entity in entities:
                crt_frame = self.df.iloc[:, (self.df.columns.get_level_values(0).isin(variables[key])) & (self.df.columns.get_level_values(1) == entity)]
                crt_frame[key, entity] = crt_frame.sum(axis=1,skipna = False)
                self.df = pd.merge(self.df, crt_frame[key, entity], left_index = True, right_index = True, how = 'left')

    def ReduceVariableDimension(self, suffix, variables, entities=None, type='pca'):
        """
        Applies dimension reduction to reduce given N variables into
        one-dimensional variable. Currently only PCA available.
        """
        if entities == None:
            entities = list(np.unique(self.df.columns.get_level_values(1).values))

        for entity in entities:
            frame = self.df.iloc[:, (self.df.columns.get_level_values(0).isin(variables)) & (self.df.columns.get_level_values(1)==entity)]
            frame = frame.dropna(axis = 0, how = 'any')
            # Check that all variables are included for given entity
            if set(variables).issubset(frame.columns.get_level_values(0)) == False:
                raise MyException("Not all variables present for " + entity + ".")       
            if type == 'pca':
                from sklearn.decomposition import PCA
                dimreductor = PCA(n_components = 1)
            else:
                raise MyException("Invalid dimension reduction tehnique.")
            
            frame_red = pd.DataFrame(dimreductor.fit_transform(frame))
            frame_red.columns = [suffix]
            frame_red.columns = pd.MultiIndex.from_tuples(list(zip(frame_red.columns,[entity]*len(frame_red.columns))))
            frame_red.index = frame.index
            self.df = pd.merge(self.df, frame_red, left_index = True, right_index = True, how = 'left')
            