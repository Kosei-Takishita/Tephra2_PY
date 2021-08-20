import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns
from matplotlib.font_manager import FontProperties
from scipy import stats

direc1 = ''  # please fill if you want to save at the specific directory
erupt = pd.DataFrame([[21001, 2000, "2021/1/1 0:00", 1000],
                      [21002, 4000, "2021/1/2 0:00", 2000]],
                     columns=["erno", "ejecta", "ertime",
                              "h_p"])  # read_csv('inpfile/forwarderuptlist.csv', header=0, index_col=0)
erupt["ertime"] = pd.to_datetime(erupt["ertime"])
# set the load in each combination of h and v
fontfilename = ''
fp = FontProperties(fname=fontfilename)
topo_calc = pd.DataFrame(
    np.concatenate([np.array(np.meshgrid(np.arange(-3000, 3000, 300), np.arange(-2400, 2400, 300))).reshape(2, -1),
                    -9999 * np.ones((1, 320))]).T,
    columns=["x", "y", "h"])  # pd.read_csv('inpfile/forwardpoint-cross300m_forcalc.csv', header=0)
# demfilename = '../14-DEM/SakuraDEM.csv'  made from GSI DEM data
dem = pd.DataFrame([[-9999, -9999, -9999, -9999, -9999],
                    [-9999, -9999, 5, 5, -9999],
                    [-9999, -9999, 5, 5, -9999],
                    [-9999, 10, 20, 15, 5],
                    [5, 20, 40, 30, 10],
                    [-9999, 5, 10, 5, -9999],
                    [-9999, -9999, -9999, -9999, -9999]])  # pd.read_csv(demfilename, header=0, index_col=0)
wind_col = np.array(np.meshgrid(["h", "vx", "vy"],
                                ["g", 999, 975, 950, 925, 900, 875, 850, 825, 800, 774, 748,
                                 724, 700, 679, 658, 638, 619, 600, 582, 565, 548, 531, 515, 500])).reshape(2, -1).T
winddat_MSM = pd.DataFrame(np.zeros((2, 75)),
                           index=["2018/4/30 12:00", "2018/4/30 15:00"],
                           columns=wind_col)  # pd.read_csv(windfilename, header=[0, 1], index_col=0)
winddat_MSM.index = pd.to_datetime(winddat_MSM.index)
parsivel = pd.DataFrame(columns=['h', 'd', 'x', 'y'],
                        data=[[10, 500, 400, 300]],
                        index=['site0'])
# [m asl, m from the vent, m to the East from the vent, m to the North from the vent]


def wind(erno):
    mh = erupt.loc[erno, "mh"]
    hr = erupt.loc[erno, 'ertime'].hour
    time1 = erupt.loc[erno, 'ertime'] - dt.timedelta(hours=hr + 9 - hr // 3 * 3,
                                                     minutes=erupt.loc[
                                                         erno, 'ertime'].minute)  # +9 to hr means JST->UTC
    time0 = time1 - dt.timedelta(hours=3)
    time2 = time1 + dt.timedelta(hours=3)
    time3 = time1 + dt.timedelta(hours=6)
    h0 = np.array([winddat_MSM.loc[time0, (slice(None), 'h')].sort_index(ascending=False),
                   winddat_MSM.loc[time1, (slice(None), 'h')].sort_index(ascending=False),
                   winddat_MSM.loc[time2, (slice(None), 'h')].sort_index(ascending=False),
                   winddat_MSM.loc[time3, (slice(None), 'h')].sort_index(ascending=False)]).T
    vx0 = np.array([winddat_MSM.loc[time0, (slice(None), 'vx')].sort_index(ascending=False),
                    winddat_MSM.loc[time1, (slice(None), 'vx')].sort_index(ascending=False),
                    winddat_MSM.loc[time2, (slice(None), 'vx')].sort_index(ascending=False),
                    winddat_MSM.loc[time3, (slice(None), 'vx')].sort_index(ascending=False)]).T
    vy0 = np.array([winddat_MSM.loc[time0, (slice(None), 'vy')].sort_index(ascending=False),
                    winddat_MSM.loc[time1, (slice(None), 'vy')].sort_index(ascending=False),
                    winddat_MSM.loc[time2, (slice(None), 'vy')].sort_index(ascending=False),
                    winddat_MSM.loc[time3, (slice(None), 'vy')].sort_index(ascending=False)]).T
    # There are 25 layers in the original script
    hf = [interp1d([0, 1, 2, 3], h0[i], kind="cubic") for i in range(25)]
    vxf = [interp1d([0, 1, 2, 3], vx0[i], kind="cubic") for i in range(25)]
    vyf = [interp1d([0, 1, 2, 3], vy0[i], kind="cubic") for i in range(25)]
    h_intp = [hf[i]((erupt.loc[erno, 'ertime'] - time0) / (time1 - time0) - 3) + 0 for i in range(25)]
    vx_intp = [vxf[i]((erupt.loc[erno, 'ertime'] - time0) / (time1 - time0) - 3) + 0 for i in range(25)]
    vy_intp = [vyf[i]((erupt.loc[erno, 'ertime'] - time0) / (time1 - time0) - 3) + 0 for i in range(25)]
    vx100 = [interp1d(h_intp, vx_intp)(1000) * (i + 1) * 0.1 for i in range(10)] + [interp1d(h_intp, vx_intp)(
        1100 + i * 100) + 0 for i in range(mh // 100)]
    vy100 = [interp1d(h_intp, vy_intp)(1000) * (i + 1) * 0.1 for i in range(10)] + [interp1d(h_intp, vy_intp)(
        1100 + i * 100) + 0 for i in range(mh // 100)]
    h100 = [i * 100 for i in range(mh // 100 + 10)]
    return h100, vx100, vy100


def output_300m(erno):
    # 放出量の初期値と風の場と下降流を与えて降灰量（総量，速度ごと）を算出する
    # 高度毎，速度ごとに一様
    mh = erupt.loc[erno, "mh"]
    shokichi = pd.DataFrame(columns=range(1100, mh + 1100, 100),
                            index=[0.0250, 0.0433, 0.0559, 0.0662, 0.0751, 0.0830, 0.0902, 0.0969, 0.106, 0.119, 0.137,
                                   0.162, 0.187, 0.212, 0.237, 0.275, 0.325, 0.375, 0.425, 0.475, 0.550, 0.650, 0.750,
                                   0.850, 0.931, 1.25, 1.74, 2.32, 2.98, 3.72, 4.99, 6.97])
    vt = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.3, 1.5, 1.7, 1.9, 2.2, 2.6, 3,
                   3.4, 3.8, 4.4, 5.2, 6, 6.8, 7.6, 8.8, 10.4, 12, 13.6, 15.2, 17.6, 20.8])
    cood = pd.DataFrame()
    w_rate = pd.DataFrame()
    tpoint = pd.DataFrame()
    winddat = pd.read_csv("wind" + str(erno) + "_100m.csv", index_col=None, header=0)
    [g, mu, rho_a, rho_p] = [9.81, 1.8e-5, 1.293, 2640]
    h_p = np.unique(topo_calc["h"]).astype(int)
    K = 100  # (m^2/s)
    FTT = 3600  # (s)
    C = 2.5 * K / FTT ** 1.5
    for vt_i in vt:
        print(vt_i, dt.datetime.now())
        d_p = shokichi.index[np.where(vt == vt_i)[0]][0] / 1000
        coordinate = pd.DataFrame(index=winddat["h"].values, columns=["t", "vz", "vx", "vy", "x", "y"])
        coordinate["vx"] = winddat["x"].values
        coordinate["vy"] = winddat["y"].values

        def vz(h):
            rho_ah = rho_a * np.exp(-h * 100 / 8200)
            vz1 = (g * d_p ** 2 * (rho_p - rho_ah) / 18 / mu)
            if rho_ah * vz1 * d_p / mu < 6:
                vz = vz1
            else:
                vz2 = (d_p * (4 * g ** 2 * (rho_p - rho_ah) ** 2 / 225 / rho_ah / mu) ** (1 / 3))
                if (rho_ah * vz2 * d_p / mu < 500) & (rho_ah * vz2 * d_p / mu >= 6):
                    vz = vz2
                else:
                    vz3 = (3.1 * g * d_p * (rho_p - rho_ah) / rho_ah) ** 0.5
                    vz = vz3
            return vz

        coordinate.iloc[:, 1] = list(map(vz, range(mh // 100 + 10)))
        coordinate.iloc[:, 0] = 100 / coordinate.iloc[:, 1]
        coordinate = coordinate.sort_index()
        for i in range(len(coordinate)):
            coordinate.iloc[i, 4:] = [np.dot(coordinate.iloc[:i + 1, 0], coordinate.iloc[:i + 1, 2]),
                                      np.dot(coordinate.iloc[:i + 1, 0], coordinate.iloc[:i + 1, 3])]
        output_newx = pd.DataFrame(columns=shokichi.columns, index=h_p)
        output_newy = pd.DataFrame(columns=shokichi.columns, index=h_p)
        traj_x = interp1d(coordinate.index, coordinate["x"], fill_value="extrapolate")
        traj_y = interp1d(coordinate.index, coordinate["y"], fill_value="extrapolate")
        t_seg = 0.0032 * (shokichi.columns - 1000) ** 2 / K
        # [2] By calculating the difference between the coordinates at the segregation height and the coordinates at the
        # surface site altitude, the coordinates of the center of tephra dispersion segregating from each height to each
        # site altitude are obtained and tabulated.
        for h_seg in shokichi.columns:
            output_newx[h_seg] = coordinate.iloc[h_seg // 100 - 1, 4] - traj_x(h_p - 100)
            output_newy[h_seg] = coordinate.iloc[h_seg // 100 - 1, 5] - traj_y(h_p - 100)

        def w1(i):
            n = int(topo_calc.loc[i, "h"])
            dx = topo_calc.loc[i, "x"] - output_newx.loc[n, :str(mh + 1000)]
            dy = topo_calc.loc[i, "y"] - output_newy.loc[n, :str(mh + 1000)]
            t = np.array([np.sum(coordinate.iloc[(n // 100):rise, 0]) - (n % 100) /
                          coordinate.iloc[(n // 100) + 1, 0] for rise in range(10, len(coordinate))])
            if np.max(t + t_seg) < 3600:
                sigma = 4 * K * (t + t_seg)
            else:
                sigma = np.where(t + t_seg < 3600, 4 * K * (t + t_seg), 1.6 * C * (t + t_seg) ** 2.5)
            # np.exp(-30) = 9.3e-18. if smaller than this, The load cannot be larger than 1e-6
            if np.max(np.array(-(dx ** 2 + dy ** 2) / sigma) > -40):
                w0 = 1 / (sigma * 3.141592) * np.exp(np.array(-(dx ** 2 + dy ** 2) / sigma)) * 1000  # [t -> kg]
            else:
                w0 = np.zeros(len(t))
            return w0

        def t_slice(i):
            n = int(topo_calc.loc[i, "h"])
            t = np.array([np.sum(coordinate.iloc[(n // 100):rise, 0]) - (n % 100) /
                          coordinate.iloc[(n // 100) + 1, 0] for rise in range(10, len(coordinate))])
            t_sum = np.round((t + t_seg) / 60, 2)
            return t_sum

        output_neww = pd.DataFrame(list(map(w1, topo_calc.index)), index=topo_calc.index)
        output_newx["category"] = "traj_x"
        output_newx["value1"] = d_p
        output_newx["value2"] = output_newx.index
        output_newy["category"] = "traj_y"
        output_newy["value1"] = d_p
        output_newy["value2"] = output_newy.index
        output_newx.columns = output_newx.columns.astype(str)
        output_newy.columns = output_newy.columns.astype(str)
        cood = cood.append(output_newx)
        cood = cood.append(output_newy)
        w_rate = w_rate.append(output_neww)
        # [3] load concentration every settling velocity interval. Estimating loads of each site and every v_t interval
        # using 2-D gaussian formula, and then make a table
        output_newt = pd.DataFrame(list(map(t_slice, topo_calc.index)))
        tpoint = tpoint.append(output_newt)
    w_rate["index"] = np.repeat(vt,
                                len(topo_calc))  # np.unique(bestfit[bestfit["OO"] > 0]["v"].values), len(point_calc))
    tpoint["index"] = np.repeat(vt,
                                len(topo_calc))  # np.unique(bestfit[bestfit["OO"] > 0]["v"].values), len(point_calc))
    w_rate = w_rate.reset_index().set_index("index")
    tpoint = tpoint.reset_index().set_index("index")
    w_rate.to_csv(direc1 + "w_rate_er" + str(erno) + "_Tephra2_300m.csv")
    tpoint.to_csv(direc1 + "tpoint_er" + str(erno) + "_Tephra2_300m.csv")


def calash_300m(erno):
    # [4] loop every tephra segregation profile: we made 91 options but use only 3 here
    st = dt.datetime.now()
    mh = erupt.loc[erno, "mh"]
    ejection = erupt.loc[erno, "ejecta"]
    w_rate = pd.read_csv(direc1 + "w_rate_er" + str(erno) + "_Tephra2_300m.csv", index_col=0)
    tpoint = pd.read_csv(direc1 + "tpoint_er" + str(erno) + "_Tephra2_300m.csv", index_col=0)
    shokichi = pd.DataFrame(columns=range(1100, mh + 1100, 100), index=np.unique(w_rate.index))
    vt = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.3, 1.5, 1.7, 1.9, 2.2, 2.6, 3, 3.4, 3.8,
          4.4, 5.2, 6, 6.8, 7.6, 8.8, 10.4, 12, 13.6, 15.2, 17.6, 20.8]
    obs = pd.read_csv(direc1 + "obs_er" + str(erno) + "_time_mfilt.csv")
    vtd = np.sum(obs.loc[:, "0.05":"20.8"].fillna(0).values, axis=0) / np.sum(
        obs.loc[:, "0.05":"20.8"].fillna(0).values)

    def w3(vt):
        w3 = pd.DataFrame(columns=["w1[kg/m2]", "vt[m/s]", "h_seg", "point", "seg", "time[min]"])
        for seg in [40, 49, 90]:
            # seg = int(bestfit.loc[vt, "seg"])
            if seg < 40:
                cum = [stats.norm.cdf(1000 / mh * j, loc=seg % 10, scale=2 ** int(seg // 10 - 1)) for j in
                       range(int(mh / 100 + 1))]
                den = [cum[j + 1] - cum[j] for j in range(int(mh / 100))]
                for i in range(len(shokichi.columns)):
                    shokichi.iloc[:, i] = den[i] * ejection * np.array(vtd)
            elif (seg >= 40) & (seg < 80):
                if seg % 10 < 5:
                    cum = [stats.lognorm.cdf(1000 / mh * j, 1, loc=seg % 10, scale=seg // 10 - 3) for j in
                           range(int(mh / 100 + 1))]
                    den = [cum[j + 1] - cum[j] for j in range(int(mh / 100))]
                    for i in range(len(shokichi.columns)):
                        shokichi.iloc[:, i] = den[i] * ejection * np.array(vtd)
                else:
                    cum = [stats.lognorm.cdf(1000 / mh * j, 1, loc=10 - seg % 10, scale=seg // 10 - 3) for j in
                           range(int(mh / 100 + 1))]
                    den = [cum[-j - 1] - cum[-j - 2] for j in range(int(mh / 100))]
                    for i in range(len(shokichi.columns)):
                        shokichi.iloc[:, i] = den[i] * ejection * np.array(vtd)
            elif (seg >= 80) & (seg < 90):
                if seg % 10 == 0:
                    den = [2 / int(mh / 100) * (10 - 1000 / mh * j) / 10 for j in range(int(mh / 100))]
                    for i in range(len(shokichi.columns)):
                        shokichi.iloc[:, i] = den[i] * ejection * np.array(vtd)
                else:
                    den = [2 / int(mh / 100) * min(1000 / mh * j / (seg % 10), (10 - 1000 / mh * j) / (10 - (seg % 10)))
                           for
                           j in range(int(mh / 100))]
                    for i in range(len(shokichi.columns)):
                        shokichi.iloc[:, i] = den[i] * ejection * np.array(vtd)
            elif seg == 90:
                for i in shokichi.columns:
                    shokichi.loc[:, i] = ejection / len(shokichi.columns) * np.array(vtd)
            else:
                print("seg is not accurate" + str(seg))
            w_rate_vt = w_rate[w_rate.index == vt].iloc[:, 1:].fillna(0)
            tpoint_vt = tpoint[tpoint.index == vt].iloc[:, 1:].fillna(0)
            res = np.array(w_rate_vt * shokichi.loc[vt].values)
            result = pd.DataFrame(res.reshape(-1).T, columns=["w1[kg/m2]"])
            result = result[result > 10 ** -6].fillna(0)
            result = (result * 10 ** 8).astype(int) / 10 ** 8
            result["vt[m/s]"] = vt
            result["h_seg"] = np.tile(w_rate_vt.columns.astype(int), (1, len(result) // len(w_rate_vt.columns)))[
                                  0].T * 100 + 1000
            result["point"] = np.tile(w_rate[w_rate.index == vt].iloc[:, 0], (len(result) // len(w_rate_vt),
                                                                              1)).T.reshape(len(result), 1)
            result["seg"] = seg
            result["time[min]"] = np.array(tpoint_vt.fillna(0)).reshape(len(result), 1)
            result.columns = result.columns.astype(str)
            result.index = result["point"]
            w3 = w3.append(result[result["w1[kg/m2]"] > 0])
        return w3

    weight = pd.concat(list(map(w3, vt)))
    weight.to_csv(direc1 + "weight3_er" + str(erno) + "_Tephra2_cross_300m.csv", index=False)
    en = dt.datetime.now()
    print(en - st)


x_int = 10.546
y_int = 12.325
x = np.arange(-7354, 1342 * x_int - 7354, x_int)  # generating the drawing range of x
y = np.arange(862 * y_int - 4521, -4521, -y_int)  # generating the drawing range of y
X, Y = np.meshgrid(x, y)


def ashmap(erno):
    caldat = pd.read_csv(direc1 + "weight3_er" + str(erno) + "_Tephra2_cross_300m.csv", index_col=None)
    vt = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.3, 1.5, 1.7, 1.9, 2.2, 2.6, 3, 3.4, 3.8,
          4.4, 5.2, 6, 6.8, 7.6, 8.8, 10.4, 12, 13.6, 15.2, 17.6, 20.8]
    cal_sheet = pd.DataFrame(index=range(len(topo_calc)), columns=vt)
    for vt_i in vt:
        for seg in [40, 49, 90]:
            for p in cal_sheet.index:
                sakura = caldat[(caldat["point"] == p) & (caldat["vt[m/s]"] == vt_i) & (caldat["seg"] == seg)][
                    "w1(p,h,v)[kg/m2]"]
                if len(sakura) == 0:
                    cal_sheet.loc[p, vt_i] = 1e-6
                else:
                    cal_sheet.loc[p, vt_i] = sakura.sum()
            plt.close()
            plt.figure(figsize=(4, 3), dpi=200)
            sns.set_context('paper')
            plt.contourf(X, Y, dem, levels=[0, 2000], colors=['0.9'], linewidths=0.7)
            plt.contourf(np.array(topo_calc["x"]).reshape(48, 35), np.array(topo_calc["y"]).reshape(48, 35),
                         np.log10(list(cal_sheet[vt_i])).reshape(48, 35), levels=[-5.999, -5, -4, -3, -2, -1, 0],
                         colors=["#0074BD", "#2EBEEC", "#AFE0F0", "#F3EFC6", "#F7BF95", "#E8746F", "#B03547"],
                         linewidths=0.5)
            plt.contour(X, Y, dem, levels=range(200, 1100, 200), colors=['0.4'], linewidths=0.5)
            plt.contour(X, Y, dem, levels=[0], colors=['k'], linewidths=0.5)
            plt.title(
                erupt.loc[erno, 'ertime'].strftime(' %y/%m/%d %H:%M') + ' vt=' + str(vt_i) + 'm/s K=100 seg=' + str(
                    int(seg)), fontproperties=fp, fontsize=12)
            plt.vlines(0, -4521, 10611 - 4521, color='0.6', linewidth=0.5)
            plt.hlines(0, -7354, 14142 - 7354, color='0.6', linewidth=0.5)
            plt.xticks([(i - 3) * 2000 for i in range(7)], [-6, -4, -2, 0, 2, 4, "km"], fontproperties=fp)
            plt.xlim([-7354, 14142 - 7354])
            plt.yticks([(i - 2) * 2000 for i in range(5)], ["km", -2, 0, 2, 4], fontproperties=fp)
            plt.ylim([-4521, 10611 - 4521])
            plt.savefig(
                direc1 + 'map/' + str(erno) + '_seg' + str(int(seg)) + 'Tephra2_' + str(int(vt_i * 100)) + '.jpg',
                bbox_inches='tight')


for erno in [21001, 21002]:
    h, vx, vy = wind(erno)
    pd.DataFrame(np.array([h, vx, vy]).T, columns=["h", "x", "y"]).to_csv("wind" + str(erno) + "_100m.csv")
    output_300m(erno)
    calash_300m(erno)
    ashmap(erno)
