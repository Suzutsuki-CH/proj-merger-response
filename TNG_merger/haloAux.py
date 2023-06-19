import illustris_python as il
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import cm
import tqdm

class HaloObj: # Class of Halo Objects
    
    def __init__(self, hID, sID, bPATH, mkdir=False):
        """
        """
        self.haloID = hID
        self.snapshotID = sID
        self.basePath = bPATH

        fields = ['Masses','Coordinates','Velocities']
        self.gas = il.snapshot.loadHalo(self.basePath, self.snapshotID, self.haloID, 'gas', fields=fields)

        fields = ['Coordinates','Velocities']
        self.DM = il.snapshot.loadHalo(self.basePath, self.snapshotID, self.haloID, 'dm', fields=fields)

        fields = ['Masses','Coordinates','Velocities']
        self.star = il.snapshot.loadHalo(self.basePath, self.snapshotID, self.haloID, 'star', fields=fields)

        halo = il.groupcat.loadSingle(self.basePath, self.snapshotID, haloID=self.haloID)
        self.CMV = halo['GroupVel']
        self.CM = halo['GroupCM']

        self.R200 = halo['Group_R_TopHat200']  
        self.mkdir = mkdir
        
        if self.mkdir == True:
            target_PATH = "s%s_h%s" % (self.snapshotID, self.haloID)
            isExist = os.path.exists(target_PATH)
            print(isExist)
            if not isExist:
                os.makedirs(target_PATH)
                print("New directory made")
    

    def calculate_relative(self, particleType: str):

        if particleType == "DM":
            relaV = self.DM['Velocities'] - self.CMV
            relaR = self.DM['Coordinates'] - self.CM

            return relaV, relaR
        
        elif particleType == "gas":
            relaV = self.gas['Velocities'] - self.CMV
            relaR = self.gas['Coordinates'] - self.CM

            return relaV, relaR
        
        elif particleType == "star":
            relaV = self.star['Velocities'] - self.CMV
            relaR = self.star['Coordinates'] - self.CM

            return relaV, relaR
        
        else:
            print('No such particle')


    def claculate_radial_v(self, particleType):
        relaV, relaR = self.calculate_relative(particleType)

        radialv = []
        radialv_vec = []

        for i, v in enumerate(relaV):
            rv = np.inner(v, relaR[i])/np.linalg.norm(relaR[i])
            radialv.append(rv)
            radialv_vec.append(rv * relaR[i]/np.linalg.norm(relaR[i]))
            
        radialv = np.array(radialv)
        radialv_vec = np.array(radialv_vec)

        return radialv, radialv_vec


    def r_vr(self, particleType: str):
        radialv, radialv_vec = self.claculate_radial_v(particleType)
        relaV, relaR = self.calculate_relative(particleType)

        Radius = np.linalg.norm(relaR, axis=1)

        fig, ax = plt.subplots(1,2, figsize=(12,5), dpi=300, sharey=True)

        ax[0].hist2d(Radius/np.max(Radius), radialv, bins=150);
        ax[0].set_xlabel("r/r$_{max}$", fontsize=15)
        ax[0].set_ylabel("v$_r$", fontsize=17)
        ax[0].axvline(x=self.R200/np.max(Radius),
                    color='red',ls="--", lw=3, alpha=0.4, label="Virial Radius")
        ax[0].legend()

        ax[1].hist2d(np.log10(Radius/np.max(Radius)), radialv, bins=150);
        ax[1].axvline(x=np.log10(self.R200/np.max(Radius)),
                    color='red',ls="--", lw=3, alpha=0.4, label="Virial Radius")
        ax[1].set_xlabel("log(r/r$_{max}$)", fontsize=15)
        
        return fig, ax
    

    def binning(self, particleType, bin_num, axis=2):
        relaV, relaR = self.calculate_relative(particleType)
        min_values = np.min(relaR, axis=0)
        max_values = np.max(relaR, axis=0)
        area = max_values - min_values

        # sample_ind = (np.random.randint(0, len(max_values), sampleSize))
        # relaR = relaR[sample_ind]
        # relaV = relaV[sample_ind]

        if axis == 0:
            relaR = np.array([relaR[:,1], relaR[:,2]]).T
            area = np.array([area[1], area[2]])
            initialR = np.array(min_values[1], min_values[2])
        elif axis == 1:
            relaR = np.array([relaR[:,0], relaR[:,2]]).T
            area = np.array([area[0], area[2]])
            initialR = np.array(min_values[0], min_values[2])
        elif axis == 2:
            relaR = np.array([relaR[:,0], relaR[:,1]]).T
            area = np.array([area[0], area[1]])
            initialR = np.array(min_values[0], min_values[1])
        else: 
            pass

        binsize = area / bin_num
        print(binsize)
        print(relaR.shape)
        
        vert_velo = []
        centerP = []
        meanvelo = []

        for i in tqdm.tqdm(range(bin_num[0])):
            for j in range(bin_num[1]):
                mm = initialR + np.array([i*binsize[0], j*binsize[1]])
                MM = initialR + np.array([(i+1)*binsize[0], (j+1)*binsize[1]])
                centerP.append((mm+MM)/2)
                
                mask = np.all((relaR >= mm) & (relaR < MM), axis=1)
                particle_in_bin = relaV[mask]
                # print(len(particle_in_bin))

                binned_V = np.mean(particle_in_bin, axis=0)

                if axis == 0:
                    meanvelo.append([binned_V[1], binned_V[2]])
                    vert_velo.append(binned_V[0])
                elif axis == 1:
                    meanvelo.append([binned_V[0], binned_V[2]])
                    vert_velo.append(binned_V[1])
                elif axis == 2:
                    meanvelo.append([binned_V[0], binned_V[1]])
                    vert_velo.append(binned_V[2])
                    
                else: pass

        centerP = np.array(centerP)
        meanvelo = np.array(meanvelo)
        vert_velo = np.array(vert_velo)


        return centerP, meanvelo, vert_velo, binsize
    

    def veloPlot(self, particleType, bin_num, axis=2):
        c, mv, vv, bs = self.binning(particleType, bin_num)
        
        norm_factor = np.max(bs)
        limC = np.max(np.abs(c)) + norm_factor

        mv_normed = mv/np.nanmax(np.linalg.norm(mv, axis=1)) * norm_factor * 1.5
        
        cw = cm.get_cmap('coolwarm', 24)
        vv_color = (vv - np.nanmin(vv))/np.nanmax(vv-np.nanmin(vv))

        plt.figure(figsize=(10,10), dpi=300)
        ax = plt.gca()
        ax.set_facecolor("silver")
        for i, p in tqdm.tqdm(enumerate(c)):
            x = (p - mv_normed[i]/2)[0]
            y = (p - mv_normed[i]/2)[1]
            dx = mv_normed[i][0]
            dy = mv_normed[i][1]
            #print(x,y,dx,dy)
            ax.set_xlim(-limC, limC)
            ax.set_ylim(-limC, limC)
            #plt.quiver(x,y,dx,dy, color="black", alpha=0.7);
            ax.plot([x,x+dx], [y,y+dy], color=cw(vv_color)[i], lw=1.05)
            ax.scatter(x+dx, y+dy, marker=".", s=15, color="black")
            ax.grid()

        ax.set_title("S%s H%s, axis=%s, particle=%s" % (self.snapshotID, self.haloID, axis, particleType) , fontsize=20)
        ax.set_xlabel('$\Delta$x [ckpc/h]', fontsize = 15)
        ax.set_ylabel('$\Delta$y [ckpc/h]', fontsize = 15)