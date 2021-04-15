import numpy as np
import pandas as pd
import datetime as dt
from scipy.interpolate import interp1d
from scipy import stats

# directory
direc1 = ''  # if you want to save or read files in another directory, please set

# input parameters
# wind

winddat_MSM = pd.DataFrame([[0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]],
                           index=["2018/4/30 12:00", "2018/4/30 15:00"],
                           columns=[[999, "h"], [999, "vx"], [999, "vy"], ["g", "h"], ["g", "vx"],
                                    ["g", "vy"]])  # pd.read_csv(windfilename, header=[0, 1], index_col=0)
winddat_MSM.index = pd.to_datetime(winddat_MSM.index)
# site
validplist = pd.DataFrame([["erno0", "2021/1/1 0:00", "site0", "site1", np.nan, np.nan],
                           ["erno1", "2021/1/2 0:00", "site0", "site1", "site2",
                            np.nan]])  # pd.read_csv("", header=None, index_col=0)
parsivel = pd.DataFrame(columns=['h', 'd', 'x', 'y'],
                        data=[[10, 500, 400, 300]],
                        index=[
                            'site0'])  # [m asl, m from the vent, m to the East from the vent, m to the North from the vent]
# eruption code, ejecta (t), occuring time, plume height (m agl)
erupt = pd.DataFrame([[21001, 2000, "2021/1/1 0:00", 1000],
                      [21002, 4000, "2021/1/2 0:00", 2000]],
                     columns=["erno", "ejecta", "ertime", "h_p"])  # pd.read_csv('', header=0, index_col=0)
erupt["ertime"] = pd.to_datetime(erupt["ertime"])

g = 9.81  # m/s^2
mu = 1.8e-5  # Pa s
rho_a = 1.293  # kg/m^3
rho_p = 2640  # kg/m^3
K = 100  # m^2/s
# mapping dem
# point map... elevation data of sites for calculation, dem...elevation data of sites for drawing coastlines and contours 
point_map = pd.DataFrame(
    np.concatenate([np.array(np.meshgrid(np.arange(-3000, 3000, 300), np.arange(-2400, 2400, 300))).reshape(2, -1),
                    np.ones((1, 320))]).T,
    columns=["x", "y", "h"])  # pd.read_csv('', header=0) The real data is made from the GSI database
demfilename = '../14-DEM/SakuraDEM.csv'  # made from GSI DEM data
dem = pd.DataFrame([[-9999, -9999, -9999, -9999, -9999],
                    [-9999, -9999, 5, 5, -9999],
                    [-9999, 10, 20, 15, 5],
                    [5, 20, 40, 30, 10],
                    [-9999, 5, 10, 5, -9999],
                    [-9999, -9999, -9999, -9999, -9999]])  # pd.read_csv(demfilename, header=0, index_col=0)
vent_elevation = 1000  # (m asl)


def wind(erno):
    # interpolate wind data every 3 hours to get the wind at time when the eruption occurred
    h_p = erupt.loc[erno, "h_p"]
    hr = erupt.loc[erno, 'ertime'].hour
    time1 = erupt.loc[erno, 'ertime'] - dt.timedelta(hours=hr + 9 - hr // 3 * 3,
                                                     minutes=erupt.loc[erno, 'ertime'].minute)  # +9 means JST->UTC
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
    h = [hf[i]((erupt.loc[erno, 'ertime'] - time0) / (time1 - time0) - 3) + 0 for i in range(25)]
    vx = [vxf[i]((erupt.loc[erno, 'ertime'] - time0) / (time1 - time0) - 3) + 0 for i in range(25)]
    vy = [vyf[i]((erupt.loc[erno, 'ertime'] - time0) / (time1 - time0) - 3) + 0 for i in range(25)]
    vx100 = [interp1d(h, vx)(vent_elevation) * (i + 1) * 0.1 for i in range(10)] + \
            [interp1d(h, vx)(vent_elevation + 100 + i * 100) + 0 for i in range(h_p // 100)]
    vy100 = [interp1d(h, vy)(vent_elevation) * (i + 1) * 0.1 for i in range(10)] + \
            [interp1d(h, vy)(vent_elevation + 100 + i * 100) + 0 for i in range(h_p // 100)]
    h100 = [i * 100 for i in range(h_p // 100 + 10)]
    return h100, vx100, vy100


x_int = 10.546  # x-axis resolution (m)
y_int = 12.325  # y-axis resolution (m)
x = np.arange(0, 1342 * x_int, x_int)  # drawing range of x-axis from 0 to 1341
y = np.arange(861 * y_int, -y_int, -y_int)  # drawing range of y-axis from 0 to 860
X, Y = np.meshgrid(x, y)


def output(erno):
    K = 100
    # input ejecta and wind and calculate tephra-fall load every v_t interval
    p_valid = validplist.loc[erno, 2:][validplist.loc[erno, 2:] == validplist.loc[erno, 2:]].values
    point = parsivel.loc[p_valid, :]
    h_p = np.unique(point["h"]).astype(int)
    h_p = erupt.loc[erno, "h_p"]
    shokichi = pd.DataFrame(columns=range(vent_elevation + 100, h_p + vent_elevation + 100, 100),
                            index=[0.0250, 0.0433, 0.0559, 0.0662, 0.0751, 0.0830, 0.0902, 0.0969, 0.106, 0.119, 0.137,
                                   0.162, 0.187, 0.212, 0.237, 0.275, 0.325, 0.375, 0.425, 0.475, 0.550, 0.650, 0.750,
                                   0.850, 0.931, 1.25, 1.74, 2.32, 2.98, 3.72, 4.99, 6.97])
    cood = pd.DataFrame()
    l_rate = pd.DataFrame()
    tpoint = pd.DataFrame()
    winddat = pd.read_csv("wind" + str(erno) + "_100m.csv", index_col=None, header=0)
    C = 2.5 * K / 3600 ** 1.5
    # 落下速度毎ループ
    for d in shokichi.index:
        d_p = d / 1000
        # [1] calculating horizontal coordinate with respect to the location at the elevation of 10m asl every layer
        # border and get coodinates at the elevation of segregation height and site location by interpolation
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

        coordinate.iloc[:, 1] = list(map(vz, range(h_p // 100 + 10)))
        coordinate.iloc[:, 0] = 100 / coordinate.iloc[:, 1]
        coordinate = coordinate.sort_index()
        for i in range(len(coordinate)):
            coordinate.iloc[i, 4:] = [np.dot(coordinate.iloc[:i + 1, 0], coordinate.iloc[:i + 1, 2]),
                                      np.dot(coordinate.iloc[:i + 1, 0], coordinate.iloc[:i + 1, 3])]
        output_newx = pd.DataFrame(columns=shokichi.columns, index=h_p)
        output_newy = pd.DataFrame(columns=shokichi.columns, index=h_p)
        traj_x = interp1d(coordinate.index, coordinate["x"], fill_value="extrapolate")
        traj_y = interp1d(coordinate.index, coordinate["y"], fill_value="extrapolate")
        t_seg = 0.0032 * (shokichi.columns - vent_elevation) ** 2 / K
        # [2] By calculating the difference between the coordinates at the segregation height and the coordinates at the
        # surface site altitude, the coordinates of the center of tephra dispersion segregating from each height to each
        # site altitude are obtained and tabulated.
        for h_seg in shokichi.columns:
            output_newx[h_seg] = list(map(lambda h: coordinate.iloc[h_seg // 100 - 1, 4] - traj_x(h - 100), h_p))
            output_newy[h_seg] = list(map(lambda h: coordinate.iloc[h_seg // 100 - 1, 5] - traj_y(h - 100), h_p))

        def w1(i):
            n = int(point.loc[i, "h"])
            dx = point.loc[i, "x"] - output_newx.loc[n, :str(h_p + vent_elevation)]
            dy = point.loc[i, "y"] - output_newy.loc[n, :str(h_p + vent_elevation)]
            t = np.array([np.sum(coordinate.iloc[(n // 100):rise, 0]) - (n % 100) /
                          coordinate.iloc[(n // 100) + 1, 0] for rise in range(10, len(coordinate))])
            if np.max(t + t_seg) < 3600:
                sigma = 4 * K * (t + t_seg)
            else:
                sigma = np.where(t + t_seg < 3600, 4 * K * (t + t_seg), 1.6 * C * (t + t_seg) ** 2.5)
            # np.exp(-30) = 9.3e-18. if smaller than this, The load cannot be larger than 1e-6
            if np.max(np.array(-(dx ** 2 + dy ** 2) / sigma) > -40):
                w0 = 1 / (sigma * 3.141592) * np.exp(np.array(-(dx ** 2 + dy ** 2) / sigma)) * 1000  # (t -> kg)
            else:
                w0 = np.zeros(len(t))
            return w0

        def t_slice(i):
            n = int(point.loc[i, "h"])
            t = np.array([np.sum(coordinate.iloc[(n // 100):rise, 0]) - (n % 100) /
                          coordinate.iloc[(n // 100) + 1, 0] for rise in range(10, len(coordinate))])
            t_sum = np.round((t + t_seg) / 60, 2)
            return t_sum

        output_neww = pd.DataFrame(list(map(w1, point.index)), index=point.index)
        output_newx["category"] = "traj_x"
        output_newx["value1"] = d
        output_newx["value2"] = output_newx.index
        output_newy["category"] = "traj_y"
        output_newy["value1"] = d
        output_newy["value2"] = output_newy.index
        output_newx.columns = output_newx.columns.astype(str)
        output_newy.columns = output_newy.columns.astype(str)
        cood = cood.append(output_newx)
        cood = cood.append(output_newy)
        l_rate = l_rate.append(output_neww)
        # [3] load concentration every settling velocity interval. Estimating loads of each site and every v_t interval
        # using 2-D gaussian formula, and then make a table
        output_newt = pd.DataFrame(list(map(t_slice, point.index)))
        tpoint = tpoint.append(output_newt)
    l_rate["index"] = np.repeat(shokichi.index, len(point))
    tpoint["index"] = np.repeat(shokichi.index, len(point))
    l_rate = l_rate.reset_index().set_index("index")
    tpoint = tpoint.reset_index().set_index("index")
    l_rate.to_csv(direc1 + "l_rate_er" + str(erno) + "_Tephra2PY_sites.csv")
    tpoint.to_csv(direc1 + "tpoint_er" + str(erno) + "_Tephra2PY_sites.csv")


def calash(erno):
    l_rate = pd.read_csv(direc1 + "l_rate_er" + str(erno) + "_Tephra2PY_sites.csv", index_col=0)
    tpoint = pd.read_csv(direc1 + "tpoint_er" + str(erno) + "_Tephra2PY_sites.csv", index_col=0)
    h_p = erupt.loc[erno, "h_p"]
    shokichi = pd.DataFrame(columns=range(vent_elevation + 100, h_p + vent_elevation + 100, 100),
                            index=np.unique(l_rate.index))
    load = pd.DataFrame()
    ejection = erupt.loc[erno, "ejecta"]
    vtd = [1 / len(shokichi) for i in range(len(shokichi))]
    # [4] loop every tephra segregation profile: 91 options
    for seg in range(91):
        if seg < 40:
            cum = [stats.norm.cdf(1000 / h_p * j, loc=seg % 10, scale=2 ** (seg // 10 - 1)) for j in
                   range(int(h_p / 100 + 1))]
            den = [cum[j + 1] - cum[j] for j in range(int(h_p / 100))]
            for i in range(len(shokichi.columns)):
                shokichi.iloc[:, i] = den[i] * ejection * np.array(vtd)
        elif (seg >= 40) & (seg < 80):
            if seg % 10 < 5:
                cum = [stats.lognorm.cdf(1000 / h_p * j, 1, loc=seg % 10, scale=seg // 10 - 3) for j in
                       range(int(h_p / 100 + 1))]
                den = [cum[j + 1] - cum[j] for j in range(int(h_p / 100))]
                for i in range(len(shokichi.columns)):
                    shokichi.iloc[:, i] = den[i] * ejection * np.array(vtd)
            else:
                cum = [stats.lognorm.cdf(1000 / h_p * j, 1, loc=10 - seg % 10, scale=seg // 10 - 3) for j in
                       range(int(h_p / 100 + 1))]
                den = [cum[-j - 1] - cum[-j - 2] for j in range(int(h_p / 100))]
                for i in range(len(shokichi.columns)):
                    shokichi.iloc[:, i] = den[i] * ejection * np.array(vtd)
        elif (seg >= 80) & (seg < 90):
            if seg % 10 == 0:
                den = [2 / int(h_p / 100) * (10 - 1000 / h_p * j) / 10 for j in range(int(h_p / 100))]
                for i in range(len(shokichi.columns)):
                    shokichi.iloc[:, i] = den[i] * ejection * np.array(vtd)
            else:
                den = [2 / int(h_p / 100) * min(1000 / h_p * j / (seg % 10), (10 - 1000 / h_p * j) / (10 - (seg % 10)))
                       for j in range(int(h_p / 100))]
                for i in range(len(shokichi.columns)):
                    shokichi.iloc[:, i] = den[i] * ejection * np.array(vtd)
        else:
            for i in shokichi.columns:
                shokichi.loc[:, i] = ejection / len(shokichi.columns) * np.array(vtd)
        for vt in shokichi.index:
            l_rate_vt = l_rate[l_rate.index == vt].iloc[:, 1:].fillna(0)
            tpoint_vt = tpoint[tpoint.index == vt].iloc[:, 1:].fillna(0)
            res = np.array(l_rate_vt * shokichi.loc[vt].values)
            result = pd.DataFrame(res.reshape(res.shape[0] * res.shape[1]).T, columns=["w1(p,h,v)[kg/m2]"])
            result = result[result > 10 ** -6].fillna(0)
            result = (result * 10 ** 8).astype(int) / 10 ** 8
            result["vt[m/s]"] = vt
            result["h_seg"] = np.tile(l_rate_vt.columns.astype(int), (1, len(result) // len(l_rate_vt.columns)))[
                                  0].T * 100 + vent_elevation
            result["point"] = np.tile(l_rate[l_rate.index == vt].iloc[:, 0], (len(result) // len(l_rate_vt),
                                                                              1)).T.reshape(len(result), 1)
            result["seg"] = seg
            result["time[min]"] = np.array(tpoint_vt.fillna(0)).reshape(len(result), 1)
            result.columns = result.columns.astype(str)
            result.index = result["point"]
            load = load.append(result[result["w1(p,h,v)[kg/m2]"] > 0])
    load.to_csv(
        direc1 + "load3_er" + str(erno) + "_tephra2PY.csv", index=False)


for erno in [18223, 18242, 18254, 18418, 19099, 19100]:
    h, vx, vy = wind(erno)
    pd.DataFrame(np.array([h, vx, vy]).T, columns=["h", "x", "y"]).to_csv("wind" + str(erno) + "_100m.csv")
    output(erno)
    calash(erno)
