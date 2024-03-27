import qutip
import qutip_cupy

qobj = qutip.Qobj([0, 1], dtype="cupyd")
qobj.data