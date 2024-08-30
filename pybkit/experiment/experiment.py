from abc import ABC, abstractmethod
from typing import Any


class Phase(object):
    
    def __init__(self, **kw):
        self.prev_phase = (None,)
        self.next_phase = (None,)
        self.kw = {}
        self.kw.update(kw)
        self.tstart = None
        self.tlen = None
        self.initialized = False

    def setup(self, params):
        # first, check if all the previous phases are done
        # and find the longest of the start + len to compute our tstart
        tstart = 0
        for pl in self.prev_phase:
            if pl is not None:
                if pl.initialized == False:
                    return
                # if we get to this point, all the phases preceeding this one
                # have executed, and it's our turn. We compute what our start time
                # is based on the longest of the previous times + lengths
                new_start = pl.tstart + pl.tlen
                if new_start > tstart:
                    tstart = new_start
        # set our start time to this, and calculate our length
        self.tstart = tstart
        self.tlen = self.getlength(params)
        self.initialized = True
        # now try to set up the following phases:
        for pl in self.next_phase:
            if pl is not None:
                pl.setup(params)

    def getlength(self, params):
        # compute length
        print("In getlength for: ", self.__class__.__name__, self.tstart)
        return 1.0

    def dophase(self, cxn, params):
        # NB: instead of passing tstart as a parameter, here, it should now be read from
        # self.tstart
        # print("Running commands for: ", self.__repr__())
        pass

    def print_checkpoint(self):
        # print("Running commands for: {:s}".format(self.__repr__()))
        pass

    def __repr__(self):
        return "%s" % self.__class__.__name__  # , repr(self.kw) if len(self.kw) > 0 else '', self.tstart)


class SyncPhase(Phase):
    def __init__(self, **kw):
        super(SyncPhase, self).__init__(**kw)
        self.tsync = 0
        if 'tsync' in self.kw.keys():
            self.tsync = self.kw['tsync']
        self.period = None
        if 'period' in self.kw.keys():
            self.period = int(self.kw['period'])

    def setup(self, params):
        # first, check if all the previous phases are done
        # and find the longest of the start + len to compute our tstart
        tstart = 0
        for pl in self.prev_phase:
            if pl is not None:
                if not pl.initialized:
                    return None
                # if we get to this point, all the phases preceeding this one
                # have executed, and it's our turn. We compute what our start time
                # is based on the longest of the previous times + lengths
                new_start = pl.tstart + pl.tlen
                if new_start > tstart:
                    tstart = new_start
        # set our start time to this, and calculate our length
        if self.period is None:
            self.period = int(params['trapModulationTTLPeriod_100ns'])
        if self.tsync >= self.period:
            raise ValueError("In a synced phase: tsync needs to be less than period")
        sample_period_nicard = 0.1e-6
        curr_tstart_int = int(tstart / sample_period_nicard)
        curr_tsync = curr_tstart_int % self.period
        if curr_tsync > self.tsync:
            self.tstart = (curr_tstart_int + self.period - (curr_tsync - self.tsync)) * sample_period_nicard
        else:
            self.tstart = (curr_tstart_int + (self.tsync - curr_tsync)) * sample_period_nicard
        self.tlen = self.getlength(params)
        self.initialized = True
        # now try to set up the following phases:
        for pl in self.next_phase:
            if pl is not None:
                pl.setup(params)
                
    
class ExperimentPhases:
    
    def __init__(self) -> None:
        pass
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
    

class Sequential:
    
    def __init__(self) -> None:
        pass
                
                

class Experiment(ABC):
    
    def __init__(self, cxn) -> None:
        self.cxn = cxn
        self.channels = None
        self.params = None
        self.iterparams = None
    
    @abstractmethod
    def setup(self):
        pass
    
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def analyze(self):
        pass
    
    @abstractmethod
    def cleanup(self):
        pass
    
    @abstractmethod
    def save(self):
        pass
    
    
    
from collections import OrderedDict
from experiments_utils import *
from phaseDefinitions import *
from rfsoc_funcs import *
import phases
import numpy as np
    
from phases.base import *
from phases.library import *
    
    
class NuclearSpinT1(Experiment):
    
    def __init__(self, cxn) -> None:
        super().__init__(cxn)
        
    def setup(self):
        super().setup()
        self.params.update({
            'saveParams': {},
            'saveImages_photometrics': True,
            'isServo302': False,
            'isServo369': False,
            'isAlignTraps': True,
            'keepSeqLenFixed': False
        })
        self.iterparams.update(
            OrderedDict([
                ('iterator', np.arange(1) if self.params['autoRun'] else np.arange(1000)),
                ('tweezerPowerWait_mW', [300]),
                ('waitingTime', [0, 0.2, 0.4, 0.6, 0.8, 1.5]),
                ('with770', [1]),
                ('withSpinReadout', [1, 0]),
        ]))
        
    def run(self, params):
        if params['withSpinReadout'] == 0:
            array_circuit_phase_list = [wait(t=1e-6)]
        if params['with770'] == 1:
            switch_phase = [switch_on_770()]
        else:
            switch_phase = [switch_off_770()]
        self.expt = ExperimentPhases(*Sequential(
            wait(t=0.1e-6),
            Sequential(*load_image_cooling),
            Sequential(*rearrange_image),
            switch_on_649_shutter(),
            Sequential(*cooling_770on_pump_trapmod_spinpol),  # 3P0

            nuclear_spin_half_pi(),
            nuclear_spin_half_pi(),

            ramp_off_and_hold_trap_modulation(),  # 3P0 without trap modulation
            ramp_tweezers_to_value_mW(p=params['tweezerPowerWait_mW']),

            switch_on_369_shutter(),
            refTTL(t=1e-6),
            Sequential(*switch_phase),
            wait(t=params['waitingTime']),
            switch_on_770(),

            ramp_tweezers_to_value_mW(p=params['tweezerPower3P0_mW']),
            ramp_on_and_hold_trap_modulation(),
            Sequential(*array_circuit_phase_list),

            Sequential(*imaging399_depump_trapmodoff),  # 3P0
            switch_off_369_shutter(),
            switch_off_649_shutter(),
            Sequential(*quadon_770off_image_reset)
        ))
        self.expt.run(self.cxn, params)
