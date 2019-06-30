import pandas as pd
import numpy as np

class MyException(Exception):
    pass

class GenericFrame:
    """
    Generic class to store meta methods common to different classes.
    """
    def __init__(self):
        pass

    def ReturnDf(self):
        """Return df."""
        return self.df.copy()

    def ReplaceDf(self,df):
        """Replaces df."""
        self.df = df

    def PlotVariables(self, variables, entities, ax=None):
        """
        Plot given series from frame
        """
        self.VariablesCheck(variables,entities)
        self.df.iloc[:, (self.df.columns.get_level_values(0).isin(variables)) & (self.df.columns.get_level_values(1).isin(entities))].plot(ax=ax)

    def GetVariables(self, variables, entities=None):
        """
        Get given series from frame
        """
        # If no entities selected, get those for which all given variables exists
        if entities == None:
            entities = self.EntitiesDefault(variables)
        # If entities selected, check that all variables exist for them
        else:
            self.VariablesCheck(variables,entities)

        frame = self.df.iloc[:, (self.df.columns.get_level_values(0).isin(variables)) & (self.df.columns.get_level_values(1).isin(entities))].copy()
        # Unused levels must be dropped as pandas default bahaviour does not do this
        frame.columns = frame.columns.remove_unused_levels()
        return frame

    def DropVariables(self, variables, entities=None):
        """
        Drop variables from frame in-place.
        """

        # If no entities selected, get those for which all given variables exists
        if entities == None:
            entities = self.EntitiesDefault(variables)
        # If entities selected, check that all variables exist for them
        else:
            self.VariablesCheck(variables,entities)

        from itertools import product as iterprod
        self.df.drop(list(iterprod(variables, entities)), axis=1, inplace=True)

        # Unused levels must be dropped as pandas default bahaviour does not do this
        self.df.columns = self.df.columns.remove_unused_levels()

    def KeepVariables(self, variables, entities=None):
        """
        Drop other than designated variables from frame in-place.
        """
        from itertools import product as iterprod

        # If no entities selected, get those for which all given variables exists
        if entities == None:
            entities = self.EntitiesDefault(variables)
        # If entities selected, check that all variables exist for them
        else:
            self.VariablesCheck(variables,entities)

        all_columns = list(zip(self.df.columns.get_level_values(0), self.df.columns.get_level_values(1)))
        keep_columns = list(iterprod(variables, entities))
        drop_cols = [item for item in all_columns if item not in keep_columns]
        self.df.drop(drop_cols, axis=1, inplace=True)
        self.df.columns = self.df.columns.remove_unused_levels()        

class MTSDataModel(GenericFrame):
    """
    Class for data storing and maniplations data in multi-variate
    time series format.
    """
    def __init__(self, filepath, colnames):
        """
        Read in data in csv format from path filepath.
        Input colnames is a dict that dictates column names.
        """
        super(MTSDataModel, self).__init__()

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
    def ExpandingSampleCalc(fun,frame,minobsamount,escKwargDict):
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
            crtresultframe = fun(crtframe, **escKwargDict)
            
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
                # Force to multi-column. We get from above merge one warning but in rest of loops it dissappears thanks to this
                resultframefull.columns = pd.MultiIndex.from_tuples([(x[0]+"_"+d, x[1]) for x in crtresultframe.columns])                

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
        frame = frame.iloc[:, (frame.columns.get_level_values(0).isin([var])) & (frame.columns.get_level_values(1)==entity)]
        # Truncate NAs from single variable, needed for mra function to run as it cannot
        # handle NA values
        frame = frame.dropna(axis = 0, how = 'any')
        # Apply wavelet mra
        pandas2ri.activate()
        mraseries = wvltpckg.WaveletMRA(frame,levels=levels, filter=filter)
        mraseries = pandas2ri.ri2py_dataframe(mraseries)
        mraseries.columns = [var+'_'+x for x in mraseries.columns]
        mraseries.index = frame.index
        # Modify to multi-columned
        mraseries.columns = pd.MultiIndex.from_tuples(list(zip(mraseries.columns,[entity]*len(mraseries.columns))))              
        return mraseries

    @staticmethod
    def HPFilter(frame,entity,var,lamb):
        """
        Function for HP filter.
        """
        from statsmodels.tsa.filters import hp_filter
        frameout = pd.DataFrame({'a': hp_filter.hpfilter(frame.values, lamb)[0], 'b':hp_filter.hpfilter(frame.values, lamb)[1]})
        frameout.index = frame.index
        frameout.columns = pd.MultiIndex.from_tuples(list(zip([var+"_hpcy",var+"_hptr"],[entity]*len(frameout.columns))))
        return frameout

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

    ####################
    # Meta methods
    ####################

    def FilterApply(self,variables,entities,ffun,expanding,minobsamount,faKwargs):
        """
        Wrapper method to apply different filter calculations.

        variables   : List of variable names to be filtered. If multiple variable names,
                      variables under same entity will be trubcated to same length before filtering.
        entities    : List of entity names.
        ffun        : Filter method/functions.
        expanding   : == 'none' (default), calculates decomposition on full sample
                      == 'expanding', calculates decomposition on expanding sample and stores expanding series
                      == 'redfull', calculates decomposition on expanding sample and stores reduced full series
                      == 'both', calculates decomposition on expanding sample stores both expanding and reduced full series
        minobsamount: Int, minumum amount of observations in sample. If number of observations less than minobsamount,
                      then no filtering will be performed. In expanding sample calculations uses  minobsamount observations
                      in pre-sample.
        faKwargs    : dict of inputs for ffun

        """
        for entity in entities:            
            # Select variables under given entity into frame
            frame = self.df.iloc[:, (self.df.columns.get_level_values(0).isin(variables)) & (self.df.columns.get_level_values(1)==entity)].copy()
            
            # Truncate NAs (if any) for all variables under given entity, both from start and end
            frame = frame.dropna(axis = 0, how = 'any')
            
            # Check that we have at least minobsamount observations in full sample. If not, skip entity.
            if len(frame) < minobsamount:
                print("For entity {} number of observations {} less than minobsamount = {}.\nNo filtering performed!\n-------"
                    .format(entity, len(frame),minobsamount))
                continue
            
            # Loop over variables
            for var in variables:
                escKwargDict = faKwargs.copy()
                escKwargDict['entity'] = entity; escKwargDict['var'] = var 

                # If decomposition on using full sample
                if expanding=='none':
                    # Apply MRA decomposition per each variable
                    resultframe = ffun(frame,**escKwargDict)
                    self.df = pd.merge(self.df, resultframe, left_index = True, right_index = True, how = 'left')
                # If decomposition using expanding sample, storing only expanding
                elif expanding=='expanding':
                    # Apply MRA decomposition to expanding sample per each variable
                    resultframe,_ = self.ExpandingSampleCalc(ffun, frame, minobsamount,escKwargDict)
                    self.df = pd.merge(self.df, resultframe, left_index = True, right_index = True, how = 'left')
                # If decomposition using expanding sample, storing only reduced full samples
                elif expanding=='redfull':
                    # Apply MRA decomposition to expanding sample per each variable
                    _,resultframe = self.ExpandingSampleCalc(ffun, frame, minobsamount,escKwargDict)
                    self.df = pd.merge(self.df, resultframe, left_index = True, right_index = True, how = 'left')

                # If decomposition using expanding sample, storing both expanding and reduced full samples
                elif expanding=='both':
                    # Apply MRA decomposition to expanding sample per each variable
                    resultframeexp,resultframefull = self.ExpandingSampleCalc(ffun, frame, minobsamount,escKwargDict)
                    self.df = pd.merge(self.df, resultframeexp,  left_index = True, right_index = True, how = 'left')
                    self.df = pd.merge(self.df, resultframefull, left_index = True, right_index = True, how = 'left')
                # else throw error
                else:
                    raise MyException("FilterApply: Invalid selection for argument expanding.")        

    ####################
    # Check methods
    ####################
    def EntitiesDefault(self,variables):
        """
        Helper function to extract entities for which 
        all chosen variables exist.
        """
        entities = []
        # All possible entities
        all_entities = list(np.unique(self.df.columns.get_level_values(1).values))
        # Loop over all_entities
        for crtentity in all_entities:
            # If variables belong to list of all variables under current entity,
            # append current entity to entities
            if set(variables).issubset(list(self.df.iloc[:, self.df.columns.get_level_values(1).isin([crtentity])].columns.get_level_values(0))):
                entities.append(crtentity)

        # If no entities found, this means no variables found. Return error.
        if len(entities) == 0:
            raise MyException("EntitiesDefault: Not all variables present for given entities list.") 
        else:
            return entities        

    def VariablesCheck(self,variables,entities):
        """Helper function to check chosen variables exist for chosen entities."""
        for entity in entities:
            frame = self.df.iloc[:, (self.df.columns.get_level_values(0).isin(variables)) & (self.df.columns.get_level_values(1) == entity)].copy()       
            if set(variables).issubset(frame.columns.get_level_values(0)) == False:
                raise MyException("VariablesCheck: Not all required variables present for " + entity + ".")            

    ####################
    # Operation methods
    ####################
    def DeflateVariables(self, variables, entities=None, infvar = 'Inflation'):
        """
        For selected 1st level variables and 2nd level entities, deflate chosen
        variables with chosen inflation variable. 
        
        Defaults to using all 2nd level entities and variable "Inflation" as
        inflation variable.

        Adds deflated series into df with suffix '_def'.
        """

        # If no entities selected, get those for which all given variables exists
        if entities == None:
            entities = self.EntitiesDefault(variables)
        # If entities selected, check that all variables exist for them
        else:
            self.VariablesCheck(variables,entities)

        # Check that inflation variable exists for entities
        self.VariablesCheck([infvar],entities)

        for seriestodefl in variables:
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
            
    def DetrendVariables(self, variables, dttype = 'ld1', entities=None):
        """
        Collection of de-trending operations. Options (dttype) are:
          - ldN: Nth log-differences
          - sdN: Nth difference
          - lg : simple natural logs
        
        Appends de-trended series to data frame with suffix dttype.
        """
        # If no entities selected, get those for which all given variables exists
        if entities == None:
            entities = self.EntitiesDefault(variables)
        # If entities selected, check that all variables exist for them
        else:
            self.VariablesCheck(variables,entities)

        for seriestodt in variables:    
            crt_frame = self.df.iloc[:, (self.df.columns.get_level_values(0).isin([seriestodt])) & (self.df.columns.get_level_values(1).isin(entities))].copy()
            if dttype[:2] == 'ld':
                crt_frame = np.log(crt_frame).diff(int(dttype[2:3]))
            elif dttype[:2] == 'sd':
                crt_frame = crt_frame.diff(int(dttype[2:3]))
            elif dttype[:2] == 'lg':
                crt_frame = np.log(crt_frame)
            else:
                raise MyException("Invalid dttype selected.")

            crt_frame.rename(columns={seriestodt:seriestodt+"_"+dttype}, inplace=True, level = 0)
            self.df = pd.merge(self.df, crt_frame, left_index = True, right_index = True, how = 'left')
        
    def MRADecomposition(self, variables, entities=None, levels=6, filter='la8', minobsamount=40, expanding='none'):
        """
        Wavelet multi-resolution decomposition of given variables.

        For each variable/entity/sample end date combination, appends level + 1 new variables to
        data frame corresponding to MODWT MRA details and smooth.
        """
        
        # If no entities selected, get those for which all given variables exists
        if entities == None:
            entities = self.EntitiesDefault(variables)
        # If entities selected, check that all variables exist for them
        else:
            self.VariablesCheck(variables,entities)        

        # Get R wavelet functions
        wvltpackage = self.ConstructWvltFunctions()

        # Apply wavelet MRA filtering using function MRADecompose
        faKwargs = {'wvltpckg':wvltpackage, 'levels':levels, 'filter':filter}     
        self.FilterApply(variables=variables, entities=entities, ffun=self.MRADecompose, expanding=expanding, minobsamount=minobsamount, faKwargs=faKwargs)

    def HPFiltering(self, variables, entities=None, lamb=1600, minobsamount=40, expanding='none'):
        """
        For each variable/entity/sample end date combination, appends 2 new HP variables to
        data frame: cycle (_hpcy) and trend (_hptr) .        
        """
        
        # If no entities selected, get those for which all given variables exists
        if entities == None:
            entities = self.EntitiesDefault(variables)
        # If entities selected, check that all variables exist for them
        else:
            self.VariablesCheck(variables,entities)        

        # Apply wavelet MRA filtering using function self.HPFilter
        faKwargs = {'lamb':lamb}
        self.FilterApply(variables=variables, entities=entities, ffun=self.HPFilter, expanding=expanding, minobsamount=minobsamount, faKwargs=faKwargs)

    def ShiftVariables(self, variables, entities=None, shift = -1):
        """
        Lead or lag for given variables, controlled by shift (int). 
        When shift is positive (negative), lags (leads) variable by given 
        period amount.
        
        Appends de-trended series to data frame with suffix "_lead/lagN".
        For example, "_lead2"
        """
        # If no entities selected, get those for which all given variables exists
        if entities == None:
            entities = self.EntitiesDefault(variables)
        # If entities selected, check that all variables exist for them
        else:
            self.VariablesCheck(variables,entities)

        # Shift selected variables, append suffix to column names
        crt_frame = self.df.iloc[:, (self.df.columns.get_level_values(0).isin(variables)) & (self.df.columns.get_level_values(1).isin(entities))].copy()
        crt_frame.columns = crt_frame.columns.remove_unused_levels()
        crt_frame = crt_frame.shift(periods=shift)
        suffix = '_lag'+str(abs(shift)) if shift>=0 else '_lead'+str(abs(shift))
        crt_frame.columns.set_levels([x+suffix for x in crt_frame.columns.levels[0]], level=0, inplace = True)

        # Merge
        self.df = pd.merge(self.df, crt_frame, left_index = True, right_index = True, how = 'left')        
                 
    def SumVariables(self,variables,name,entities=None):
        """
        Aggregate given variables, under same entity, using simple sum.

        variables is a dict with key being new variable name and value designating
        variables to be summed together.
        """
        # If no entities selected, get those for which all given variables exists
        if entities == None:
            entities = self.EntitiesDefault(variables)
        # If entities selected, check that all variables exist for them
        else:
            self.VariablesCheck(variables,entities)

        for entity in entities:
            crt_frame = self.df.iloc[:, (self.df.columns.get_level_values(0).isin(variables)) & (self.df.columns.get_level_values(1) == entity)]
            crt_frame[name, entity] = crt_frame.sum(axis=1,skipna = False)
            self.df = pd.merge(self.df, crt_frame[name, entity], left_index = True, right_index = True, how = 'left')

    def ReduceVariableDimension(self, suffix, variables, entities=None, type='pca'):
        """
        Applies dimension reduction to reduce given N variables into
        one-dimensional variable. Currently only PCA available.
        """

        # If no entities selected, get those for which all given variables exists
        if entities == None:
            entities = self.EntitiesDefault(variables)
        # If entities selected, check that all variables exist for them
        else:
            self.VariablesCheck(variables,entities)

        for entity in entities:
            frame = self.df.iloc[:, (self.df.columns.get_level_values(0).isin(variables)) & (self.df.columns.get_level_values(1)==entity)]
            frame = frame.dropna(axis = 0, how = 'any')

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
