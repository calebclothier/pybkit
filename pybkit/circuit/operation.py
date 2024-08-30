from abc import ABC, abstractmethod
import numpy as np



class Operation(ABC):
    """Parent class for operations applied to ArrayCircuit objects. Only intended to be subclassed.

    All subclasses must implement the following functions:

    `Operation.__init__(arr,...)`
        Constructor which takes the instance of ArrayCircuit as its first argument

   ` Operation.get_duration()`
        Return duration of operation in seconds

    `Operation.do()`
        Execute the operation on the instance of ArrayCircuit in self.array
   """

    def __init__(self):
        self.duration = None

    def get_duration(self):
        return self.duration

    @abstractmethod
    def do(self):
        pass


class MoveOperation(Operation):

    def __init__(self):
        super(MoveOperation, self).__init__()


class WaitOperation(Operation):

    def __init__(self, duration=0):
        super(WaitOperation, self).__init__()
        self.duration = duration

    def __repr__(self):
        return f'Wait for {self.duration:.2f}'

    def do(self):
        pass


class GateOperation(Operation):
    """Parent class for gate operations acting on ArrayCircuit objects. Only intended to be subclassed.

    In addition to functions needed for any Operation, GateOperation subclasses must also implement:

        Operation.acts_on()
            Returns a list of qubits that are acted on by the gate.

    For single-qubit gates, the default do() may be sufficient, as it just acts self.fn() on the qubits
    from self.acts_on(). For two-qubit gates, a different do() may be overloaded.
   """

    def __init__(self):
        super(GateOperation, self).__init__()
        self.arr = None
        self.fn = None

    @abstractmethod
    def acts_on(self):
        pass

    def do(self):
        qubits = self.acts_on()
        if len(qubits) > 0:
            self.arr.circuit.append(cirq.Moment(self.fn(x) for x in qubits))


class GlobalGate(GateOperation):
    """ Apply a gate to all qubits in arr.

        GlobalGate(arr,fn,during)
            Global gate on ArrayCircuit arr implemented by cirq function fn. The operation takes time specified by duration.

    Note: gate is assumed to be a single-qubit gate.
    """

    def __init__(self, arr, fn, duration=0):
        super(GlobalGate, self).__init__()
        self.arr = arr
        self.fn = fn
        self.duration = duration

    def __repr__(self):
        return f'GlobalGate {self.fn}'

    def acts_on(self):
        """ Return list of qubit objects that the gate acts on. If multi-qubit gate, in tuples.
        Raises error if invalid. """
        qubits = []
        for sg in self.arr.site_groups:
            for s in sg.sites:
                if s.atom is not None:
                    if s.atom.qubit is not None:
                        qubits.append(s.atom.qubit)


class LocalGate(GlobalGate):
    """ Apply a gate to qubits at specific sites.

        LocalGate(arr,fn,coords)
            Apply gate specified by fn to qubits in arr at coordinates specified by coords.

    Note: gate is assumed to be single-qubit. Could generalize to 2QB or create 2QB version where coords are paired.
    """

    def __init__(self, arr, fn, coords, duration=0):
        super(LocalGate, self).__init__(arr, fn, duration)
        self.coords = coords  # list of xy pairs

    def __repr__(self):
        return f'LocalGate {self.fn} on {len(self.coords)} sites'

    def _in_coords(self, x):
        for c in self.coords:
            if np.all(x == c):
                return True
        return False

    def acts_on(self):
        """ Return list of qubit objects that the gate acts on. If multi-qubit gate, in tuples.
        Raises error if invalid. """
        qubits = []
        for sg in self.arr.site_groups:
            for s in sg.sites:
                if s.atom is not None:
                    if self._in_coords(s.get_coords()):
                        if s.atom.qubit is not None:
                            qubits.append(s.atom.qubit)
        return qubits
