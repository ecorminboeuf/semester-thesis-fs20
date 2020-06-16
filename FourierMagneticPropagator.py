r"""The WaveBlocks Project

This file contains the Fourier Magnetic Propagator class. The wavefunction
:math:`\Psi` is propagated in time with a splitting of the
exponential :math:`\exp(-\frac{i}{\varepsilon^2} \tau H)`.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2016 R. Bourquin
@license: Modified BSD License
"""

from numpy import array, complexfloating, dot, exp, eye, zeros, shape
from numpy.fft import fftn, ifftn
from scipy.linalg import expm

from WaveBlocksND.BlockFactory import BlockFactory
from WaveBlocksND.Propagator import Propagator
from WaveBlocksND.KineticOperator import KineticOperator
from WaveBlocksND.MagneticField import MagneticField
from WaveBlocksND.SplittingParameters import SplittingParameters

__all__ = ["FourierMagneticPropagator"]


class FourierMagneticPropagator(Propagator, SplittingParameters):
    r"""This class can numerically propagate given initial values :math:`\Psi(x_0, t_0)` on
    a potential hyper surface :math:`V(x)`, in presence of a magnetic field. The propagation is done with a splitting
    of the time propagation operator :math:`\exp(-\frac{i}{\varepsilon^2} \tau H)`.
    Available splitting schemes are implemented in :py:class:`SplittingParameters`.
    """

    def __init__(self, parameters, potential, initial_values):
        r"""Initialize a new :py:class:`FourierMagneticPropagator` instance. Precalculate the
        the kinetic operator :math:`T_e` and the potential operator :math:`V_e`
        used for time propagation.

        :param parameters: The set of simulation parameters. It must contain at least
                           the semi-classical parameter :math:`\varepsilon` and the
                           time step size :math:`\tau`.
        :param potential: The potential :math:`V(x)` governing the time evolution.
        :type potential: A :py:class:`MatrixPotential` instance.
        :param initial_values: The initial values :math:`\Psi(\Gamma, t_0)` given
                               in the canonical basis.
        :type initial_values: A :py:class:`WaveFunction` instance.

        :raise: :py:class:`ValueError` If the number of components of :math:`\Psi` does not match the
                           number of energy surfaces :math:`\lambda_i(x)` of the potential.

        :raise: :py:class:`ValueError` If the number of components of :math:`\Psi` does not match the dimension of the magnetic field :math:`\vec{B}(x)`.

        :raise: :py:class:`ValueError` If the dimensions of the splitting scheme parameters :math:`a` and :math:`b` are not equal.
        """
        # The embedded 'MatrixPotential' instance representing the potential 'V'.
        self._potential = potential

        # The initial values of the components '\psi_i' sampled at the given grid.
        self._psi = initial_values

        if self._potential.get_number_components() != self._psi.get_number_components():
            raise ValueError("Potential dimension and number of components do not match.")

        # The time step size.
        self._dt = parameters["dt"]

        # Final time.
        self._T = parameters["T"]

        # The model parameter '\varepsilon'.
        self._eps = parameters["eps"]

        # Spacial dimension d
        self._dimension = parameters["dimension"]

        # The position space grid nodes '\Gamma'.
        self._grid = initial_values.get_grid()

        # The kinetic operator 'T' defined in momentum space.
        self._KO = KineticOperator(self._grid, self._eps)

        # Exponential '\exp(-i/2*eps^2*dt*T)' used in the Strang splitting.
        # not used
        self._KO.calculate_exponential(-0.5j * self._dt * self._eps**2)
        self._TE = self._KO.evaluate_exponential_at()

        # Exponential '\exp(-i/eps^2*dt*V)' used in the Strang splitting.
        # not used
        self._potential.calculate_exponential(-0.5j * self._dt / self._eps**2)
        VE = self._potential.evaluate_exponential_at(self._grid)
        self._VE = tuple([ve.reshape(self._grid.get_number_nodes()) for ve in VE])

        # The magnetic field
        self._B = MagneticField(parameters["B"])
        # check if magnetic field and potential are of same dimension
        if self._B.get_dimension() != self._dimension:
            raise ValueError("Spacial dimension of potential and magnetic field must be the same")

        #precalculate the splitting needed
        self._a, self._b = self.build(parameters["splitting_method"])
        if shape(self._a) != shape(self._b):
            raise ValueError("Splitting scheme shapes must be the same")

        # Get inital data as function
        packet_descr = parameters["initvals"][0]
        self._initalpacket = BlockFactory().create_wavepacket(packet_descr)


    # TODO: Consider removing this, duplicate
    def get_number_components(self):
        r"""Get the number :math:`N` of components of :math:`\Psi`.

        :return: The number :math:`N`.
        """
        return self._potential.get_number_components()


    def get_wavefunction(self):
        r"""Get the wavefunction that stores the current data :math:`\Psi(\Gamma)`.

        :return: The :py:class:`WaveFunction` instance.
        """
        return self._psi


    def get_operators(self):
        r"""Get the kinetic and potential operators :math:`T(\Omega)` and :math:`V(\Gamma)`.

        :return: A tuple :math:`(T, V)` containing two ``ndarrays``.
        """
        # TODO: What kind of object exactly do we want to return?
        self._KO.calculate_operator()
        T = self._KO.evaluate_at()
        V = self._potential.evaluate_at(self._grid)
        V = tuple([v.reshape(self._grid.get_number_nodes()) for v in V])
        return (T, V)


    @staticmethod
    def _Magnus_CF4(tspan, B, N, *args):
        r"""Returns the  Fourth Order Magnus integrator :math:`\Omega(A)` according to [#]_.

        :param tspan: Full timespan of expansion.

        :param B: Magnetic field matrix :math:`B(t) = (B_{j,k}(t))_{1 \leq j, k \leq d}`.

        :param N: Number of timesteps for the expansion.

        :param *args: Additional arguments for the magnetic field :math:`B(t, *args)`

        .. [#] S. Blanes and P.C. Moan. "Fourth- and sixth-order commutator-free Magnus integrators for linear and non-linear dynamical systems". Applied Numerical Mathematics, 56(12):1519 - 1537, 2006.
        """
        # Magnus constants
        c1 = 0.5*(1.0 - 0.5773502691896258)
        c2 = 0.5*(1.0 + 0.5773502691896258)
        a1 = 0.5*(0.5 - 0.5773502691896258)
        a2 = 0.5*(0.5 + 0.5773502691896258)

        R = 1.*eye( len( B(1.*tspan[0], *args) ) )
        h = (tspan[1]-tspan[0]) / (1.*N)
        for k in range(N):
            t0 = k*h + tspan[0]
            t1 = t0 + c1*h
            t2 = t0 + c2*h
            B1 = B(t1, *args)
            B2 = B(t2, *args)
            R = dot(expm(a1*h*B1+a2*h*B2), dot(expm(a2*h*B1+a1*h*B2), R))

        return R


    def post_propagate(self, tspan):
        r"""Given an initial wavepacket :math:`\Psi_0` at time :math:`t=0`, calculate the propagated wavepacket :math:`\Psi` at time :math:`tspan \[ 0 \]`. We perform :math:`n = \lceil tspan\[ 0 \] /dt \rceil` steps of size :math:`dt`.

        :param tspan: :py class:`ndarray` consisting of end time at position 0, other positions are irrelevant.
        """

        # (ignoriere tspan[0])
        nsteps = int(tspan[0] / self._dt + 0.5)
        print("Perform " + str(nsteps) + " steps from t = 0.0 to t = " + str(tspan[0]))


        # Magnetfeld Matrix B(t)
        B = lambda t: self._B(t)

        #how many components does Psi have
        N = self._psi.get_number_components()

        #start time t_0 = 0?
        t0 = 0
        t_a = t0
        t_b = t0

        #calculate R = U(t0 + N*h, t0)
        #Use N = n_steps to account for large time difference
        t_interval = array([t0, tspan[0]])
        R = FourierMagneticPropagator._Magnus_CF4(t_interval, B, nsteps)

        # rotate the grid by the transpose of R
        self._grid.rotate(R.T)

        # Compute rotated initial data
        X = self._grid.get_nodes(flat=True)
        values = self._initalpacket.evaluate_at(X, prefactor=True)
        values = tuple([val.reshape(self._grid.get_number_nodes()) for val in values])
        self._psi.set_values(values)

        self._grid.rotate(R)

        #calculate the necessary timesteps
        for j in range(nsteps):
            for i in range(len(self._a)):
                # Integral -\int_{tspan[0]}^{tspan[1]}B^2(s)ds und zugehörige Propagation
                # (siehe Paper, Remark 3.1)
                minus_B_squared = lambda t: (-1.0) * dot(B(t), B(t))
                A = 1.0 / 8.0 * MagneticField.matrix_quad(minus_B_squared, t_a, t_a + self._a[i]*self._dt)

                X = self._grid.get_nodes(flat=True)
                VB = sum(X * dot(A, X))
                VB = VB.reshape(self._grid.get_number_nodes())
                prop = exp(-1.0j / self._eps**2 * VB) # ev. -0.5j durch -1j ersetzen...

                values = self._psi.get_values()
                values = [prop * component for component in values]

                self._potential.calculate_exponential(-1.0j *  self._a[i]*self._dt /self._eps**2)

                self._grid.rotate(R.T)
                VE = self._potential.evaluate_exponential_at(self._grid)
                self._VE = tuple([ve.reshape(self._grid.get_number_nodes()) for ve in VE])

                #apply it
                values = [self._VE * component for component in values]
                self._grid.rotate(R)

                t_interval[0] = t_b
                t_interval[1] = t_b + self._b[i]*self._dt

                U = (FourierMagneticPropagator._Magnus_CF4(t_interval, B, 1)).T
                R = dot(R , U)
                if(R.shape != U.shape):
                    raise ValueError("Shapes of R and U do not match")

                #check for obsolete splitting steps
                if(self._b[i] != 0):
                    values = [fftn(component) for component in values]

                    # Apply the kinetic operator
                    self._KO = KineticOperator(self._grid, self._eps)
                    self._KO.calculate_exponential(-0.5j * self._eps**2 * self._b[i]*self._dt)

                    TE = self._KO.evaluate_exponential_at()
                    values = [TE * component for component in values]

                    # Go back to real space
                    values = [ifftn(component) for component in values]

                #Apply
                self._psi.set_values(values)

                #update t_a and t_b
                t_a = t_a + self._a[i]*self._dt
                t_b = t_b + self._b[i]*self._dt

        return tspan[0]


    def propagate(self, tspan):
        r"""This method does nothing.
        """
