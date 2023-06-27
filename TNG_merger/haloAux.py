import illustris_python as il
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
from matplotlib import cm
import tqdm


class HaloObj:  # Class of Halo Objects
    def __init__(self, hID, sID, bPATH, mkdir=False):
        """ """
        self.haloID = hID
        self.snapshotID = sID
        self.basePath = bPATH

        fields = ["Masses", "Coordinates", "Velocities"]
        self.gas = il.snapshot.loadHalo(
            self.basePath, self.snapshotID, self.haloID, "gas", fields=fields
        )

        fields = ["Coordinates", "Velocities"]
        self.DM = il.snapshot.loadHalo(
            self.basePath, self.snapshotID, self.haloID, "dm", fields=fields
        )

        fields = ["Masses", "Coordinates", "Velocities"]
        self.star = il.snapshot.loadHalo(
            self.basePath, self.snapshotID, self.haloID, "star", fields=fields
        )

        halo = il.groupcat.loadSingle(
            self.basePath, self.snapshotID, haloID=self.haloID
        )
        self.CMV = halo["GroupVel"]
        self.CM = halo["GroupCM"]

        self.R200 = halo["Group_R_TopHat200"]
        self.mkdir = mkdir

        if self.mkdir == True:
            target_PATH = "s%s_h%s" % (self.snapshotID, self.haloID)
            isExist = os.path.exists(target_PATH)
            print(isExist)
            if not isExist:
                os.makedirs(target_PATH)
                print("New directory made")

    def period_correction(self, particleType: str):
        # This correction work only for TNG100-1 data,
        # not really sure what is the period size for other simulation
        def correction(coord):
            return ((coord + 37500) % 75000) - 37500

        if particleType == "DM":
            Rx = self.DM["Coordinates"][:, 0]
            Ry = self.DM["Coordinates"][:, 1]
            Rz = self.DM["Coordinates"][:, 2]
            if (Rx.max() - Rx.min()) > 70000:
                Rx = correction(Rx)
            if (Ry.max() - Ry.min()) > 70000:
                Ry = correction(Ry)
            if (Rz.max() - Rz.min()) > 70000:
                Rz = correction(Rz)
            self.DM["Coordinates"] = np.array([Rx, Ry, Rz]).T

        elif particleType == "gas":
            Rx = self.gas["Coordinates"][:, 0]
            Ry = self.gas["Coordinates"][:, 1]
            Rz = self.gas["Coordinates"][:, 2]
            if (Rx.max() - Rx.min()) > 70000:
                Rx = correction(Rx)
            if (Ry.max() - Ry.min()) > 70000:
                Ry = correction(Ry)
            if (Rz.max() - Rz.min()) > 70000:
                Rz = correction(Rz)
            self.gas["Coordinates"] = np.array([Rx, Ry, Rz]).T

        elif particleType == "star":
            Rx = self.star["Coordinates"][:, 0]
            Ry = self.star["Coordinates"][:, 1]
            Rz = self.star["Coordinates"][:, 2]
            if (Rx.max() - Rx.min()) > 70000:
                Rx = correction(Rx)
            if (Ry.max() - Ry.min()) > 70000:
                Ry = correction(Ry)
            if (Rz.max() - Rz.min()) > 70000:
                Rz = correction(Rz)
            self.star["Coordinates"] = np.array([Rx, Ry, Rz]).T

        else:
            print("No such particle")

    def calculate_relative(self, particleType: str):
        if particleType == "DM":
            relaV = self.DM["Velocities"] - self.CMV
            relaR = self.DM["Coordinates"] - self.CM

            return relaV, relaR

        elif particleType == "gas":
            relaV = self.gas["Velocities"] - self.CMV
            relaR = self.gas["Coordinates"] - self.CM

            return relaV, relaR

        elif particleType == "star":
            relaV = self.star["Velocities"] - self.CMV
            relaR = self.star["Coordinates"] - self.CM

            return relaV, relaR

        else:
            print("No such particle")

    def claculate_radial_v(self, particleType):
        relaV, relaR = self.calculate_relative(particleType)

        radialv = []
        radialv_vec = []

        for i, v in enumerate(relaV):
            rv = np.inner(v, relaR[i]) / np.linalg.norm(relaR[i])
            radialv.append(rv)
            radialv_vec.append(rv * relaR[i] / np.linalg.norm(relaR[i]))

        radialv = np.array(radialv)
        radialv_vec = np.array(radialv_vec)

        return radialv, radialv_vec

    def r_vr(self, particleType: str):
        radialv, radialv_vec = self.claculate_radial_v(particleType)
        relaV, relaR = self.calculate_relative(particleType)

        Radius = np.linalg.norm(relaR, axis=1)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)

        ax[0].hist2d(Radius / np.max(Radius), radialv, bins=150)
        ax[0].set_xlabel("r/r$_{max}$", fontsize=15)
        ax[0].set_ylabel("v$_r$", fontsize=17)
        ax[0].axvline(
            x=self.R200 / np.max(Radius),
            color="red",
            ls="--",
            lw=3,
            alpha=0.4,
            label="Virial Radius",
        )
        ax[0].legend()

        ax[1].hist2d(np.log10(Radius / np.max(Radius)), radialv, bins=150)
        ax[1].axvline(
            x=np.log10(self.R200 / np.max(Radius)),
            color="red",
            ls="--",
            lw=3,
            alpha=0.4,
            label="Virial Radius",
        )
        ax[1].set_xlabel("log(r/r$_{max}$)", fontsize=15)

        plt.tight_layout(pad=0.0, w_pad=-0.1, h_pad=0.3)

        return fig, ax

    def binning(self, particleType, bin_num, axis:int=2, sli:np.float64=False):
        relaV, relaR = self.calculate_relative(particleType)

        if sli:
            if axis == 0:
                slice_mask = (relaR[:,0]>-sli) &i (relaR[:,0]<sli)
            elif axis == 1:
                slice_mask = (relaR[:,1]>-sli) & (relaR[:,1]<sli)
            elif axis == 2:
                slice_mask = (relaR[:,2]>-sli) & (relaR[:,2]<sli)
            else: pass
            # print(slice_mask)
            
            relaR = relaR[slice_mask]
            relaV = relaV[slice_mask]
            
        min_values = np.min(relaR, axis=0)
        max_values = np.max(relaR, axis=0)
        area = max_values - min_values

        # sample_ind = (np.random.randint(0, len(max_values), sampleSize))
        # relaR = relaR[sample_ind]
        # relaV = relaV[sample_ind]

        if axis == 0:
            relaR = np.array([relaR[:, 1], relaR[:, 2]]).T
            area = np.array([area[1], area[2]])
            initialR = np.array([min_values[1], min_values[2]])
        elif axis == 1:
            relaR = np.array([relaR[:, 0], relaR[:, 2]]).T
            area = np.array([area[0], area[2]])
            initialR = np.array([min_values[0], min_values[2]])
        elif axis == 2:
            relaR = np.array([relaR[:, 0], relaR[:, 1]]).T
            area = np.array([area[0], area[1]])
            initialR = np.array([min_values[0], min_values[1]])
        else:
            pass

        binsize = area / bin_num
        print("binsize:", binsize)
        print(relaR.shape)
        print(initialR)

        vert_velo = []
        centerP = []
        meanvelo = []
        numP = []

        for i in tqdm.tqdm(range(bin_num[0])):
            for j in range(bin_num[1]):
                mm = initialR + np.array([i * binsize[0], j * binsize[1]])
                MM = initialR + np.array([(i + 1) * binsize[0], (j + 1) * binsize[1]])
                centerP.append((mm + MM) / 2)

                mask = np.all((relaR >= mm) & (relaR < MM), axis=1)
                particle_in_bin = relaV[mask]
                numP.append(len(particle_in_bin))
                
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

                else:
                    pass

        centerP = np.array(centerP)
        meanvelo = np.array(meanvelo)
        vert_velo = np.array(vert_velo)
        numP = np.array(numP)

        return centerP, meanvelo, vert_velo, binsize, numP
    
    def binning_fixed(self, particleType, bin_num=np.array([30,30]), frame_size=np.array([2400,2400]), axis:int=2, sli:np.float64=False):
        relaV, relaR = self.calculate_relative(particleType)

        if sli:
            if axis == 0:
                slice_mask = (relaR[:,0]>-sli) &i (relaR[:,0]<sli)
            elif axis == 1:
                slice_mask = (relaR[:,1]>-sli) & (relaR[:,1]<sli)
            elif axis == 2:
                slice_mask = (relaR[:,2]>-sli) & (relaR[:,2]<sli)
            else: pass
            # print(slice_mask)
            
            relaR = relaR[slice_mask]
            relaV = relaV[slice_mask]
            
        min_values = -0.5 * frame_size
        max_values = 0.5 * frame_size
        area = max_values - min_values

        # sample_ind = (np.random.randint(0, len(max_values), sampleSize))
        # relaR = relaR[sample_ind]
        # relaV = relaV[sample_ind]

        if axis == 0:
            relaR = np.array([relaR[:, 1], relaR[:, 2]]).T
            area = np.array([area[1], area[2]])
            initialR = np.array([min_values[1], min_values[2]])
        elif axis == 1:
            relaR = np.array([relaR[:, 0], relaR[:, 2]]).T
            area = np.array([area[0], area[2]])
            initialR = np.array([min_values[0], min_values[2]])
        elif axis == 2:
            relaR = np.array([relaR[:, 0], relaR[:, 1]]).T
            area = np.array([area[0], area[1]])
            initialR = np.array([min_values[0], min_values[1]])
        else:
            pass

        binsize = area / bin_num
        print("binsize:", binsize)
        print(relaR.shape)
        print(initialR)

        vert_velo = []
        centerP = []
        meanvelo = []
        numP = []

        for i in tqdm.tqdm(range(bin_num[0])):
            for j in range(bin_num[1]):
                mm = initialR + np.array([i * binsize[0], j * binsize[1]])
                MM = initialR + np.array([(i + 1) * binsize[0], (j + 1) * binsize[1]])
                centerP.append((mm + MM) / 2)

                mask = np.all((relaR >= mm) & (relaR < MM), axis=1)
                particle_in_bin = relaV[mask]
                numP.append(len(particle_in_bin))
                
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

                else:
                    pass

        centerP = np.array(centerP)
        meanvelo = np.array(meanvelo)
        vert_velo = np.array(vert_velo)
        numP = np.array(numP)

        return centerP, meanvelo, vert_velo, binsize, numP

    def veloGrid(self, particleType, bin_num, fixframe=False, frame_size=np.array([2400,2400]), axis=2, sli=False):
        if fixframe:
            c, mv, vv, bs, Np = self.binning_fixed(particleType, bin_num=bin_num, frame_size=frame_size, sli=sli)
        else:
            c, mv, vv, bs, Np = self.binning(particleType, bin_num, sli=sli)
        norm_factor = np.max(bs)
        limC = np.max(np.abs(c)) + norm_factor

        mv_normed = mv / np.nanmax(np.linalg.norm(mv, axis=1)) * norm_factor * 1.03

        cw = cm.get_cmap("coolwarm", 24)
        vv_color = (vv - np.nanmin(vv)) / np.nanmax(vv - np.nanmin(vv))

        plt.figure(figsize=(10, 10), dpi=300)
        ax = plt.gca()
        ax.set_facecolor("silver")
        for i, p in tqdm.tqdm(enumerate(c)):
            x = (p - mv_normed[i] / 2)[0]
            y = (p - mv_normed[i] / 2)[1]
            dx = mv_normed[i][0]
            dy = mv_normed[i][1]
            # print(x,y,dx,dy)
            ax.set_xlim(-limC, limC)
            ax.set_ylim(-limC, limC)
            # plt.quiver(x,y,dx,dy, color="black", alpha=0.7);
            ax.plot([x, x + dx], [y, y + dy], color=cw(vv_color)[i], lw=1.05)
            ax.scatter(x + dx, y + dy, marker=".", s=15, color="black")
            ax.grid()

        ax.set_title(
            "S%s H%s, axis=%s, particle=%s, slice=%s"
            % (self.snapshotID, self.haloID, axis, particleType, sli),
            fontsize=20,
        )
        ax.set_xlabel("$\Delta$x [ckpc/h]", fontsize=15)
        ax.set_ylabel("$\Delta$y [ckpc/h]", fontsize=15)
        

