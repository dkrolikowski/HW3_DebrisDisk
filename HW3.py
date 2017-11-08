import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
import pandas as pd
import scipy.interpolate as interpolate

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

wavarr = np.logspace( -1, 3, 1000 )             # Set wavelength array for whole HW in microns
nuarr  = const.c.to('micron/s').value / wavarr  # Compute frequency array from wavelength array

# Stellar properties of Fomalhaut
Tstar  = 8590
Rstar  = 1.842
Mstar  = 1.92

# Compute stellar spectrum at 10 au and 130 au
Fstar  = { '10': FluxDensity( nuarr, Tstar, Rstar, 10 ), '130': FluxDensity( nuarr, Tstar, Rstar, 130 ) }

# Plot stellar spectrum
plt.clf()
plt.loglog( wavarr, Fstar['10'], 'r-' )
plt.loglog( wavarr, Fstar['130'], 'b-' )
plt.xlim( 0.1, 100 ); plt.ylim( 1e7, 5e13 )
plt.gca().set_xticklabels([0.0,0.1,1.0,10.0,100.0])
plt.legend( ( '10 au', '130 au' ) )
plt.ylabel( 'Flux Density (Jy)' )
plt.xlabel( 'Wavelength ($\mu$m)' )
plt.savefig( 'StellarSpec.pdf' )

### Question 2 - Set up for dust Q's and power absorbed ###

class DustProperties():

    def __init__( self, Tstar, Rstar, Mstar, a, strRg, Qdict, rhog ):

        # Input quantities
        self.Tstar = Tstar # Stellar Temp in K
        self.Rstar = Rstar # Stellar Radius in Solar Radii
        self.Mstar = Mstar # Stellar Mass in Solar Masses
        self.a     = a     # Orbital distance in au
        self.strRg = strRg # Grain size string (for Qval dictionary)
        self.Qdict = Qdict # Qval dictionary
        self.rhog  = rhog  # Dust grain density in g per cc

        # Calculated quantities
        self.Rg    = float( self.strRg ) * 1e-4
        self.Qwav  = Qdict['wav']
        self.Qvals = Qdict[self.strRg]
        self.mg    = 4. / 3. * np.pi * self.Rg ** 3.0 * self.rhog

        self.dist  = 7.7 * u.pc.to('cm') # Distance to Fomalhaut

    def interpQ( self, wav ):

        # Given wavelength array, interpolates Q vals on that
        Qfit = interpolate.interp1d( self.Qwav, self.Qvals )

        return Qfit( wav )

    def PowerAbsorbed( self, wav ):

        # Calculates power absorbed by dust grain over a wavelength range in erg/s
        nu   = const.c.to('micron/s').value / wav
        flux = FluxDensity( nu, self.Tstar, self.Rstar, self.a ) / 1e23
        Qarr = self.interpQ( wav )

        return np.pi * self.Rg ** 2.0 * np.trapz( flux * Qarr, x = -nu )

    def EquilibriumTemp( self, wav ):
        # Calculates equilibrium temp of a dust grain in K

        Pabs = self.PowerAbsorbed( wav )
        Td   = ( Pabs / ( 4 * np.pi * self.Rg ** 2.0 * SBcgs ) ) ** 0.25
        
        return Td

    def EmissionSpectrum( self, wav ):
        # Calculates and returns emission spectrum of a dust grain in jy

        Td = self.EquilibriumTemp( wav )

        return np.pi * Blackbody( const.c.to('micron/s').value / wav, Td ) * self.interpQ( wav )

    def Luminosity( self, wav ):
        # Calculates luminosity of a dust grain
        
        nu   = const.c.to('micron/s').value / wav
        flux = self.EmissionSpectrum( wav ) / 1e23 # Emission spectrum in cgs

        return 4 * np.pi * self.Rg ** 2.0 * np.trapz( flux, x = -nu )

    def NumGrains( self, wav, Fdisk, wavpt ):
        # Given an observed flux value, calculates number of grains necessary to match

        Fdust = self.EmissionSpectrum( wav )

        Fdisk /= self.Rg ** 2.0 # Fdisk is already corrected for distance, must also be divided by grain size squared

        return Fdisk / interpolate.interp1d( wav, Fdust )(wavpt)

    def DiskMass( self, wav, Fdisk, wavpt ):
        # Calculates the mass in dust given number of grains in earth masses

        Ng = self.NumGrains( wav, Fdisk, wavpt )

        return self.mg * Ng * u.g.to('earthMass')

    def RadPressure( self, wav ):
        # Calculates radiation pressure force in dynes
        
        return self.PowerAbsorbed( wav ) / const.c.cgs.value

    def PRdrag( self, wav ):
        # Calculates PR drag force in dynes

        v = 2 * np.pi * np.sqrt( self.Mstar / self.a ) * ( u.au / u.yr ).to('cm/s')

        return v / const.c.cgs.value ** 2.0 * self.PowerAbsorbed( wav )

    def PRtime( self, wav ):
        # Calculates timescale over which PRdrag would work in yrs

        v  = 2 * np.pi * np.sqrt( self.Mstar / self.a ) * ( u.au / u.yr ).to('cm/s')
        P  = np.sqrt( self.a ** 3.0 / self.Mstar )

        Fg = const.G.cgs.value * self.Mstar * u.solMass.to('g') * self.mg * ( self.a * u.au.to('cm') ) ** -2.0
        B  = self.RadPressure( wav ) / Fg

        T  = P * ( B * v / const.c.cgs.value ) ** -1.0

        return T

# Read in Q values for each type of dust grain
Qdict = pd.read_csv( 'Qvalues.csv' )
rhod  = 2.0 # g per cc

# Set up class instances for each dust grain at each orbital distance
dust_tenth_10 = DustProperties( Tstar, Rstar, Mstar, 10, '0.1', Qdict, rhod )
dust_one_10   = DustProperties( Tstar, Rstar, Mstar, 10, '1', Qdict, rhod )
dust_ten_10   = DustProperties( Tstar, Rstar, Mstar, 10, '10', Qdict, rhod )
dust_perf_10  = DustProperties( Tstar, Rstar, Mstar, 10, '1000', Qdict, rhod )

dust_tenth_130 = DustProperties( Tstar, Rstar, Mstar, 130, '0.1', Qdict, rhod )
dust_one_130   = DustProperties( Tstar, Rstar, Mstar, 130, '1', Qdict, rhod )
dust_ten_130   = DustProperties( Tstar, Rstar, Mstar, 130, '10', Qdict, rhod )
dust_perf_130  = DustProperties( Tstar, Rstar, Mstar, 130, '1000', Qdict, rhod )

# Calculate power absorbed
Pabs_10 = { '0.1': dust_tenth_10.PowerAbsorbed( wavarr ), '1': dust_one_10.PowerAbsorbed( wavarr ),
             '10': dust_ten_10.PowerAbsorbed( wavarr ), '1000': dust_perf_10.PowerAbsorbed( wavarr ) }

Pabs_130 = { '0.1': dust_tenth_130.PowerAbsorbed( wavarr ), '1': dust_one_130.PowerAbsorbed( wavarr ),
             '10': dust_ten_130.PowerAbsorbed( wavarr ), '1000': dust_perf_130.PowerAbsorbed( wavarr ) }

print 'The power absorbed in erg/s for each dust grain type at 10 au is:\n'
print Pabs_10
print '\nThe power absorbed in erg/s for each dust grain type at 130 au is:\n'
print Pabs_130

## Question 3 - Emission properties of dust grain###

# Calculate equilibrium temp
Tdust_10 = { '0.1': dust_tenth_10.EquilibriumTemp( wavarr ), '1': dust_one_10.EquilibriumTemp( wavarr ),
             '10': dust_ten_10.EquilibriumTemp( wavarr ), '1000': dust_perf_10.EquilibriumTemp( wavarr ) }

Tdust_130 = { '0.1': dust_tenth_130.EquilibriumTemp( wavarr ), '1': dust_one_130.EquilibriumTemp( wavarr ),
             '10': dust_ten_130.EquilibriumTemp( wavarr ), '1000': dust_perf_130.EquilibriumTemp( wavarr ) }

print 'The equilibrium temperature in K for each dust grain type at 10 au is:\n'
print Tdust_10
print '\nThe equilibrium temperature in K for each dust grain type at 130 au is:\n'
print Tdust_130

# Calculate luminosity of dust grain
Ldust_10 = { '0.1': dust_tenth_10.Luminosity( wavarr ), '1': dust_one_10.Luminosity( wavarr ),
             '10': dust_ten_10.Luminosity( wavarr ), '1000': dust_perf_10.Luminosity( wavarr ) }

Ldust_130 = { '0.1': dust_tenth_130.Luminosity( wavarr ), '1': dust_one_130.Luminosity( wavarr ),
             '10': dust_ten_130.Luminosity( wavarr ), '1000': dust_perf_130.Luminosity( wavarr ) }

print 'The luminosity in erg/s for each dust grain type at 10 au is:\n'
print Ldust_10
print '\nThe luminosity in erg/s for each dust grain type at 130 au is:\n'
print Ldust_130

# Calculate dust grain emission spectrum
Fdust_10 = { '0.1': dust_tenth_10.EmissionSpectrum( wavarr ), '1': dust_one_10.EmissionSpectrum( wavarr ),
             '10': dust_ten_10.EmissionSpectrum( wavarr ), '1000': dust_perf_10.EmissionSpectrum( wavarr ) }

Fdust_130 = { '0.1': dust_tenth_130.EmissionSpectrum( wavarr ), '1': dust_one_130.EmissionSpectrum( wavarr ),
             '10': dust_ten_130.EmissionSpectrum( wavarr ), '1000': dust_perf_130.EmissionSpectrum( wavarr ) }

# Plot dust spectra at 10 au
plt.clf()
plt.loglog( wavarr, Fdust_10['0.1'], 'k-' )
plt.loglog( wavarr, Fdust_10['1'], 'b-' )
plt.loglog( wavarr, Fdust_10['10'], 'r-' )
plt.loglog( wavarr, Fdust_10['1000'], 'g-' )
plt.xlim( 5.0, 1000.0 ); plt.ylim( 1e4, 1e16 )
plt.gca().set_xticklabels([0.0,0.0,10.0,100.0,1000.0])
plt.legend( ( '0.1 $\mu$m', '1 $\mu$m', '10 $\mu$m', '1 mm' ), loc = 'upper right' )
plt.ylabel( 'Flux Density (Jy)' )
plt.xlabel( 'Wavelength ($\mu$m)' )
plt.savefig('DustSpectrum_10au.pdf')

# Plot dust spectra at 130 au
plt.clf()
plt.loglog( wavarr, Fdust_130['0.1'], 'k-' )
plt.loglog( wavarr, Fdust_130['1'], 'b-' )
plt.loglog( wavarr, Fdust_130['10'], 'r-' )
plt.loglog( wavarr, Fdust_130['1000'], 'g-' )
plt.xlim( 5.0, 1000.0 ); plt.ylim( 1e4, 1e16 )
plt.gca().set_xticklabels([0.0,0.0,10.0,100.0,1000.0])
plt.ylabel( 'Flux Density (Jy)' )
plt.xlabel( 'Wavelength ($\mu$m)' )
plt.legend( ( '0.1 $\mu$m', '1 $\mu$m', '10 $\mu$m', '1 mm' ), loc = 'upper right' )
plt.savefig('DustSpectrum_130au.pdf')

### Question 4 - Fomalhaut comparison ###

# From Kate Su's 2013 paper

# For the inner, hotter disk
Fdisk_30um = 5e46 / 0.1 ** 2.0

Ng_10 = { '0.1': dust_tenth_10.NumGrains( wavarr, 5e38, 30 ), '1': dust_one_10.NumGrains( wavarr, 5e38, 30 ),
             '10': dust_ten_10.NumGrains( wavarr, 5e38, 30 ), '1000': dust_perf_10.NumGrains( wavarr, 5e38, 30 ) }

Md_10 = { '0.1': dust_tenth_10.DiskMass( wavarr, 5e38, 30 ), '1': dust_one_10.DiskMass( wavarr, 5e38, 30 ),
             '10': dust_ten_10.DiskMass( wavarr, 5e38, 30 ), '1000': dust_perf_10.DiskMass( wavarr, 5e38, 30 ) }

print "The number of dust grains needed to match Fomalhaut's inner disk for each grain type at (10 au) is:\n"
print Ng_10
print "\nThe corresponding disk mass for each grain type in earth masses is:\n"
print Md_10

# For the outer, cooler disk
Fdisk_100um = 6e47 / 10.0 ** 2.0

Ng_130 = { '0.1': dust_tenth_130.NumGrains( wavarr, 6e39, 100 ), '1': dust_one_130.NumGrains( wavarr, 6e39, 100 ),
             '10': dust_ten_130.NumGrains( wavarr, 6e39, 100 ), '1000': dust_perf_130.NumGrains( wavarr, 6e39, 100 ) }

Md_130 = { '0.1': dust_tenth_130.DiskMass( wavarr, 6e39, 100 ), '1': dust_one_130.DiskMass( wavarr, 6e39, 100 ),
             '10': dust_ten_130.DiskMass( wavarr, 6e39, 100 ), '1000': dust_perf_130.DiskMass( wavarr, 6e39, 100 ) }

print "The number of dust grains needed to match Fomalhaut's outer disk for each grain type (at 130 au) is:\n"
print Ng_130
print "\nThe corresponding disk mass for each grain type in earth masses is:\n"
print Md_130

### Question 5 - Forces ###

Frad_10 = { '0.1': dust_tenth_10.RadPressure( wavarr ), '1': dust_one_10.RadPressure( wavarr ),
             '10': dust_ten_10.RadPressure( wavarr ), '1000': dust_perf_10.RadPressure( wavarr ) }

Frad_130 = { '0.1': dust_tenth_130.RadPressure( wavarr ), '1': dust_one_130.RadPressure( wavarr ),
             '10': dust_ten_130.RadPressure( wavarr ), '1000': dust_perf_130.RadPressure( wavarr ) }

print 'The radiation pressure force (in dynes) on each grain type at 10 au is:\n'
print Frad_10
print '\nThe radiation pressure force (in dynes) on each grain type at 130 au is:\n'
print Frad_130

FPR_10 = { '0.1': dust_tenth_10.PRdrag( wavarr ), '1': dust_one_10.PRdrag( wavarr ),
             '10': dust_ten_10.PRdrag( wavarr ), '1000': dust_perf_10.PRdrag( wavarr ) }

FPR_130 = { '0.1': dust_tenth_130.PRdrag( wavarr ), '1': dust_one_130.PRdrag( wavarr ),
             '10': dust_ten_130.PRdrag( wavarr ), '1000': dust_perf_130.PRdrag( wavarr ) }

print 'The PR drag force (in dynes) on each grain type at 10 au is:\n'
print FPR_10
print '\nThe PR drag force (in dynes) on each grain type at 130 au is:\n'
print FPR_130

PRtime_10 = { '0.1': dust_tenth_10.PRtime( wavarr ), '1': dust_one_10.PRtime( wavarr ),
             '10': dust_ten_10.PRtime( wavarr ), '1000': dust_perf_10.PRtime( wavarr ) }

PRtime_130 = { '0.1': dust_tenth_130.PRtime( wavarr ), '1': dust_one_130.PRtime( wavarr ),
             '10': dust_ten_130.PRtime( wavarr ), '1000': dust_perf_130.PRtime( wavarr ) }

print 'The PR drag timescale (in yrs) on each grain type at 10 au is:\n'
print PRtime_10
print '\nThe PR drag timescale (in yrs) on each grain type at 130 au is:\n'
print PRtime_130
