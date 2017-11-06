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

wavarr = np.logspace( -1, 3, 1000 )
nuarr  = const.c.to('micron/s').value / wavarr

Tstar  = 8590
Rstar  = 1.842

Fstar  = { '10': FluxDensity( nuarr, Tstar, Rstar, 10 ), '130': FluxDensity( nuarr, Tstar, Rstar, 130 ) }

plt.clf()
plt.loglog( wavarr, Fstar['10'], 'k-' )
plt.loglog( wavarr, Fstar['130'], 'b-' )
plt.xlim( 0.1, 100 ); plt.ylim( 1e7, 5e13 )
plt.legend( ( '10 au', '130 au' ) )
plt.ylabel( 'Flux Density (Jy)' )
plt.xlabel( 'Wavelength ($\mu$m)' )
plt.savefig( 'StellarSpec.pdf' )

plt.clf()
plt.loglog( wavarr, FluxDensity( nuarr, Tstar, Rstar, 7.66 * u.pc.to('au' ) ), 'k-' )
plt.xlim( 0.1, 100 )
plt.show()

### Question 2 ###

class DustProperties():

    def __init__( self, Tstar, Rstar, a, Rg, rhog ):

        self.Tstar = Tstar
        self.Rstar = Rstar
        self.a     = a
        self.Rg    = Rg
        self.rhog  = rhog

        self.mg    = 4. / 3. * np.pi * self.Rg ** 3.0 * self.rhog

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

    def EmissionSpectrum( self, wav, Qwav, Qvals ):

        Td = self.EquilibriumTemp( wav, Qwav, Qvals )

        return np.pi * Blackbody( const.c.to('micron/s').value / wav, Td ) * self.getQ( wav, Qwav, Qvals )

Qvals = pd.read_csv( 'Qvalues.csv' )

dust_tenth_10 = DustProperties( Tstar, Rstar, 10, 0.1e-4 )
dust_one_10   = DustProperties( Tstar, Rstar, 10, 1e-4 )
dust_ten_10   = DustProperties( Tstar, Rstar, 10, 10e-4 )

dust_tenth_130 = DustProperties( Tstar, Rstar, 130, 0.1e-4 )
dust_one_130   = DustProperties( Tstar, Rstar, 130, 1e-4 )
dust_ten_130   = DustProperties( Tstar, Rstar, 130, 10e-4 )

## Question 3 ###

Fdust_10 = { '0.1': dust_tenth_10.EmissionSpectrum( wavarr, Qvals['wav'], Qvals['0.1'] ),
             '1.0': dust_one_10.EmissionSpectrum( wavarr, Qvals['wav'], Qvals['1.0'] ),
             '10.0': dust_ten_10.EmissionSpectrum( wavarr, Qvals['wav'], Qvals['10.0'] ) }

Fdust_130 = { '0.1': dust_tenth_130.EmissionSpectrum( wavarr, Qvals['wav'], Qvals['0.1'] ),
             '1.0': dust_one_130.EmissionSpectrum( wavarr, Qvals['wav'], Qvals['1.0'] ),
             '10.0': dust_ten_130.EmissionSpectrum( wavarr, Qvals['wav'], Qvals['10.0'] ) }
    
plt.clf()
plt.loglog( wavarr, Fdust_10['0.1'], 'k-' )
plt.loglog( wavarr, Fdust_10['1.0'], 'b-' )
plt.loglog( wavarr, Fdust_10['10.0'], 'r-' )
plt.xlim( 5.0, 1000.0 ); plt.ylim( 1e4, 1e16 )
plt.savefig('DustSpectrum_10au.pdf')

plt.clf()
plt.loglog( wavarr, Fdust_130['0.1'], 'k-' )
plt.loglog( wavarr, Fdust_130['1.0'], 'b-' )
plt.loglog( wavarr, Fdust_130['10.0'], 'r-' )
plt.xlim( 5.0, 1000.0 ); plt.ylim( 1e4, 1e16 )
plt.savefig('DustSpectrum_130au.pdf')

### Question 4 ###
