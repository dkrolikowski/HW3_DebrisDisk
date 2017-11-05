import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
import pandas as pd
import scipy.interpolate as interpolate

import pdb

SBcgs = const.sigma_sb.cgs.value

def Blackbody( nu, T ):

    # Computes the blackbody spectrum in Jansky

    A     = 2 * const.h.cgs.value * nu ** 3.0 / const.c.cgs.value ** 2.0
    eterm = np.exp( ( const.h.cgs.value * nu ) / ( const.k_B.cgs.value * T ) ) - 1.0

    return A / eterm * 1e23

### Question 1 - Stellar Spectrum ###

def FluxDensity( nu, T, R, a ):

    # Computes flux density at a distance of a from a blackbody source
    # R is in solar radii and a is in au

    blackbody = Blackbody( nu, T )
    
    return np.pi * blackbody * ( R * u.solRad.to('au') / a ) ** 2.0

wavarr = np.logspace( -1, 2, 1000 )
nuarr  = const.c.to('micron/s').value / wavarr

Tstar  = 8590
Rstar  = 1.842

Fstar  = { '10': FluxDensity( nuarr, Tstar, Rstar, 10 ), '130': FluxDensity( nuarr, Tstar, Rstar, 130 ) }

plt.clf()
plt.loglog( wavarr, Fstar['10'], 'k-' )
plt.loglog( wavarr, Fstar['130'], 'b-' )
plt.legend( ( '10 au', '130 au' ) )
plt.ylabel( 'Flux Density (Jy)' )
plt.xlabel( 'Wavelength ($\mu$m)' )
plt.savefig( 'StellarSpec.pdf' )

### Question 2 ###

class DustProperties():

    def __init__( self, Tstar, Rstar, a, Rg ):

        self.Tstar = Tstar
        self.Rstar = Rstar
        self.a     = a
        self.Rg    = Rg

    def getQ( self, wav, Qwav, Qvals ):

        Qfit = interpolate.interp1d( Qwav, Qvals )

        return Qfit( wav )

    def PowerAbsorbed( self, wav, Qwav, Qvals ):

        nu   = const.c.to('micron/s').value / wav
        flux = FluxDensity( nu, self.Tstar, self.Rstar, self.a ) / 1e23
        Qarr = self.getQ( wav, Qwav, Qvals )

        return np.pi * self.Rg ** 2.0 * np.trapz( flux * Qarr, x = -nu )

    def EquilibriumTemp( self, wav, Qwav, Qvals ):

        Pabs = self.PowerAbsorbed( wav, Qwav, Qvals )
        Td   = ( Pabs / ( 4 * np.pi * self.Rg ** 2.0 * SBcgs ) ) ** 0.25

        return Td

Qvals = pd.read_csv( 'Qvalues.csv' )

dust_tenth = DustProperties( Tstar, Rstar, 10, 0.1e-4 )
dust_one   = DustProperties( Tstar, Rstar, 10, 1e-4 )
dust_ten   = DustProperties( Tstar, Rstar, 10, 10e-4 )

## Question 3 ###

print dust_one.EquilibriumTemp( wavarr, Qvals['wav'], Qvals['1.0'] )

wavdust = np.logspace( np.log10(5.0), 3, 1000 )
nudust  = const.c.to('micron/s').value / wavdust
    
plt.clf()
plt.loglog( wavdust, FluxDensity( nudust, Td, Rstar, Rstar ) * fit_one( wavdust ), 'k-' )
plt.show()

print Qvals['wav'].min(), Qvals['wav'].max()
