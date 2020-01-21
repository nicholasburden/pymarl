import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pymongo
import pprint
import os, sys
import numpy as np
import pandas as pd
import chart_studio as py
import chart_studio.tools as tls
import matplotlib.patches as mpatches
from sklearn import manifold, datasets


def get_mongo_db_client(conf_name, maxSevSelDelay=5000, root_dir="."):
    import yaml
    mongo_conf = None
    with open(os.path.join(root_dir, "config", "default.yaml"), 'r') as stream:
        try:
            mongo_conf = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    db_url = mongo_conf["db_url"]
    db_name = mongo_conf["db_name"]
    client = pymongo.MongoClient(db_url, ssl=True, serverSelectionTimeoutMS=maxSevSelDelay)
    return client, client[db_name]


class MongoCentral():

    def __init__(self, *args, **kwargs):
        self.conf_names = kwargs["conf_names"]
        self.root_dir = kwargs.get("root_dir")
        self.db = {}
        self._connect(self.conf_names)

    def _connect(self, conf_names):
        self.clients = {}
        for _name in conf_names:
            self.clients[_name], self.db[_name] = get_mongo_db_client(_name, root_dir=self.root_dir)

    def get_exp_names(self):
        # print("Loading keys...")
        names = []
        for key, db in self.db.items():
            query = db["runs"].distinct("config.name")  # .find({"config":None})
            names.extend(query)
            print("Done Loading...")
        return names

    def get_tag_names(self, tag, bundle=True):
        import re
        names = []
        for key, db in self.db.items():
            query = db.runs.find({"config.name": {'$regex': r'^{}\_\_(.*)'.format(tag)}},
                                 {"config.name": 1})  # .find({"config":None})
            names.extend([_q["config"]["name"] for _q in query])
            print("Done Loading...")

        if bundle:  # bundle by experiment name
            bundle_dic = {}
            for name in names:
                # tag, exp_name_time_stamp, repeat = name.split("__")
                exp_name_time_stamp = "__".join(name.split("__")[1:-1])
                exp_name = "_".join(exp_name_time_stamp.split("_")[:-1])
                if exp_name not in bundle_dic:
                    bundle_dic[exp_name] = []
                bundle_dic[exp_name].append(name)
            return bundle_dic
        return names

    def get_name_label_names(self, name, label): #, bundle=True):
        import re
        names = []
        for key, db in self.db.items(): 
            query = db.runs.find({"config.name": {'$regex': r'^{}(.*)'.format(name)},
                                  "config.label": {'$regex': r'^{}(.*)'.format(label)}},
                                 {"config.name": 1, "config.label":1})  # .find({"config":None})
            names.extend([{"name":_q["config"]["name"], "label":_q["config"]["label"], "_id":_q["_id"]} for _q in query])
            print("Done Loading...")

        #if bundle:  # bundle by experiment name
        #    bundle_dic = {}
        #    for name in names:
        #        # tag, exp_name_time_stamp, repeat = name.split("__")
        #        exp_name_time_stamp = "__".join(name.split("__")[1:-1])
        #        exp_name = "_".join(exp_name_time_stamp.split("_")[:-1])
        #        if exp_name not in bundle_dic:
        #            bundle_dic[exp_name] = []
        #        bundle_dic[exp_name].append(name)
        #    return bundle_dic
        return {"bundle1": names}
    
    def get_name_prop(self, name, props):
        if not isinstance(props, (list, tuple)):
            props = [props]
        res = []
        for key, db in self.db.items():
            if isinstance(name, dict):
                query_dct = {"_id": name["_id"]}
            else:
                query_dct = {"config.name": name}
            #print(query_dct, {prop: 1 for prop in props})
            query = db.runs.find(query_dct, {prop: 1 for prop in props})
            for _q in query:
                res.append(_q)
        return res


# Once if have downloaded a bundle of form server {instance_tags:[runs]},
# I then wish to establish the statistical properties of the averages of the properties that I am interested in
# `property_name` is the name of the property summarized across runs, `propertyT_name` is the label for the time sync signal
# if `no_fail` is set, then all runs that resulted in failures are removed from the average

def bundle_average(mongo_central,
                   items,
                   no_fails=False,
                   verbose=True,
                   window=5000,
                   t_res=10000,
                   min_steps=None):

    property_hash_lst = []
    property_lst_lst = []
    propertyT_lst_lst = []
    endT_lst_lst = []
    run_failT_lst_lst = []
    algo_label_lst = []
    for item in items:
        algo_label, bundle, property_str, propertyT_str = item
        property_hash = {}  # hash table of format {T:[properties]}
        run_failT_lst = []
        endT_lst = []

        # NOTE: In the future, would be great to check here for bundle config inconsistencies (and allow to block some entries manually TODO)
        for _instance_tag, _run_lst in bundle.items():  # iterate over all instance_tags
            for _run_id, _run_name in enumerate(_run_lst):  # iterate over all runs associate to _instance_tag
                property_res = mongo_central.get_name_prop(_run_name, "info.{}".format(property_str))
                propertyT_res = mongo_central.get_name_prop(_run_name, "info.{}".format(propertyT_str))
                fail_trace_res = mongo_central.get_name_prop(_run_name, "fail_trace")
                
                # retrieve results in form of lists
                if property_str not in property_res[0]["info"]:
                    print("Property str not available: {} not in results for run {}".format(property_str, _run_name))
                    continue
                property_lst = property_res[0]["info"][property_str]
                if propertyT_str not in propertyT_res[0]["info"]:
                    print("PropertyT str not available: {} not in results for run {}".format(propertyT_str, _run_name))
                    continue
                propertyT_lst = propertyT_res[0]["info"][propertyT_str]
                endT_lst.append(max(propertyT_lst))

                # check if fail trace is present
                has_fail_trace = ("fail_trace" in fail_trace_res[0])
                if has_fail_trace:
                    run_failT_lst.append(max(propertyT_lst))
                    if verbose:
                        print("Run (ID {}, name {}) has fail trace!".format(_run_id, _run_name))
                    if no_fails:
                        continue
                else:
                    run_failT_lst.append(float("nan"))

                if min_steps is not None and max(propertyT_lst) < min_steps:
                    continue  # do not include as not enough steps completed!

                # sort results into the results hash table
                for p, t in zip(property_lst, propertyT_lst):
                    if t in property_hash:
                        property_hash[t].append(p)
                    else:
                        property_hash[t] = [p]

        property_hash_lst.append(property_hash)
        property_lst_lst.append(property_lst)
        propertyT_lst_lst.append(propertyT_lst)
        endT_lst_lst.append(endT_lst)
        run_failT_lst_lst.append(run_failT_lst)
        algo_label_lst.append(algo_label)

    min_T = float("inf")
    max_T = -float("inf")
    for l in propertyT_lst_lst:
        if min(l) < min_T:
            min_T = min(l)
        if max(l) > max_T:
            max_T = max(l)
    Tindex_list = list(range(min_T, max_T, t_res))
    
    # TODO: create list of lists with index Tindex_list and the number of processes per bundle that are still active at this point
    # this will need to be derived from the process end times - these need to be supplied above

    propertyTindex_std_lst_lst = []
    propertyTindex_avg_lst_lst = []
    propertyTindex_min_lst_lst = []
    propertyTindex_max_lst_lst = []
    propertyTindex_cnt_lst_lst = []
    for property_hash in property_hash_lst:

        # calculate window averages
        sorted_T = sorted(property_hash.keys())
        propertyTindex_avg_lst = [0] * len(Tindex_list)
        propertyTindex_std_lst = [0] * len(Tindex_list)
        propertyTindex_min_lst = [0] * len(Tindex_list)
        propertyTindex_max_lst = [0] * len(Tindex_list)
        propertyTindex_cnt_lst = [0] * len(Tindex_list)
        for Tindex_lst_idx, Tidx in enumerate(Tindex_list):
            T_in_window = [_t for _t in sorted_T if ((_t >= Tidx) and (_t <= Tidx + window))]
            property_lst_window = []
            for t in T_in_window:
                property_lst_window.extend(property_hash[t])

                # safeguard against zero denominator
            if len(property_lst_window) > 0:
                propertyTindex_avg_lst[Tindex_lst_idx] = np.mean(np.array(property_lst_window))
                propertyTindex_std_lst[Tindex_lst_idx] = np.std(np.array(property_lst_window))
                propertyTindex_min_lst[Tindex_lst_idx] = np.min(np.array(property_lst_window))
                propertyTindex_max_lst[Tindex_lst_idx] = np.max(np.array(property_lst_window))
                propertyTindex_cnt_lst[Tindex_lst_idx] = len(property_lst_window)
            else:
                propertyTindex_avg_lst[Tindex_lst_idx] = float("nan")
                propertyTindex_std_lst[Tindex_lst_idx] = float("nan")
                propertyTindex_min_lst[Tindex_lst_idx] = float("nan")
                propertyTindex_max_lst[Tindex_lst_idx] = float("nan")
                propertyTindex_cnt_lst[Tindex_lst_idx] = 0

        propertyTindex_avg_lst_lst.append(propertyTindex_avg_lst)
        propertyTindex_std_lst_lst.append(propertyTindex_std_lst)
        propertyTindex_min_lst_lst.append(propertyTindex_min_lst)
        propertyTindex_max_lst_lst.append(propertyTindex_max_lst)
        propertyTindex_cnt_lst_lst.append(propertyTindex_cnt_lst)

    return {"uncert": propertyTindex_std_lst_lst,
            "uncert_label": "std",
            "mean": propertyTindex_avg_lst_lst,
            "mean_label": "mean",
            "count": propertyTindex_cnt_lst_lst,
            "_idx": Tindex_list,
            "endT": endT_lst_lst,
            "run_failT": run_failT_lst_lst,
            "algo_label": algo_label_lst,
            "min": propertyTindex_min_lst_lst,
            "max": propertyTindex_max_lst_lst}


def region_split(lst, idx_lst, splitsymbol):
    # print("split symbol: ", splitsymbol)
    last_symb = splitsymbol
    last_split_idx = 0
    out_lst = []
    idx_out_lst = []
    for idx, item in enumerate(lst):
        if (str(item) == str(splitsymbol) and str(item) != str(last_symb)) or (idx == len(lst) - 1):            
            region = lst[last_split_idx: idx]
            region_idxs = idx_lst[last_split_idx: idx]
            # print("NACHO!", region)
            if region != []:
                out_lst.append(region)
                idx_out_lst.append(region_idxs)
        if str(item) != str(last_symb) and str(last_symb) == str(splitsymbol):
            last_split_idx = idx
        last_symb = item
    return out_lst, idx_out_lst


def plot_bundle_avgs(bundle_avgs, figsize=(40, 20)):
    # TODO: Add skew and kurtosis plots!

    # define colors
    mean_colors = ["#6255ed", "#ff5d5d", "#038103", "#ff14ff"]*10
    uncert_colors = ["#cccbfc", "#ffcccc", "#cce5cc", "#ff14ff"]*10

    # set up figure and axes
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=20,
                           ncols=5,
                           left=0.01, # 0.00
                           right=0.7, # 0.7
                           wspace=0.01)
    ax_main = fig.add_subplot(gs[:-1, :])
    ax_main.set_xlabel("T")
    ax_main.set_ylabel("Test Win Rate")
    #ax_fail = fig.add_subplot(gs[-1, :],
                              #sharex=ax_main)

    # plot uncertainties
    for color_idx, (mean, uncert) in enumerate(zip(bundle_avgs["mean"], bundle_avgs["uncert"])):
        uncert_lower_regions, uncert_idx_regions = region_split(np.array(mean) - np.array(uncert), bundle_avgs["_idx"],
                                                                float("nan"))
        uncert_higher_regions, _ = region_split(np.array(mean) + np.array(uncert), bundle_avgs["_idx"], float("nan"))
        for uncert_lower_region, uncert_higher_region, uncert_idx_region in zip(uncert_lower_regions,
                                                                                uncert_higher_regions,
                                                                                uncert_idx_regions):
            # print(uncert_colors[color_idx % len(uncert_colors)])
            ax_main.fill_between(uncert_idx_region,
                                 uncert_lower_region,
                                 uncert_higher_region,
                                 color=uncert_colors[color_idx % len(uncert_colors)],
                                 alpha=0.4)
    """
    # plot min / max
    for color_idx, (_min, _max) in enumerate(zip(bundle_avgs["min"], bundle_avgs["max"])):
        min_regions, min_idx_regions = region_split(_min, bundle_avgs["_idx"], float("nan"))
        max_regions, max_idx_regions = region_split(_max, bundle_avgs["_idx"], float("nan"))
        # print("min/bundle_avgs:", _min, bundle_avgs["_idx"], "min_regions:", min_regions)
        # print("max/bundle_avgs:", _max, bundle_avgs["_idx"], "max_regions:", max_regions)
        for min_region, max_region, min_idx_region, max_idx_region in zip(min_regions, max_regions, min_idx_regions,
                                                                          max_idx_regions):
            ax_main.plot(min_idx_region, min_region, color=mean_colors[color_idx % len(mean_colors)], alpha=0.7,
                         linestyle="--")
            ax_main.plot(max_idx_region, max_region, color=mean_colors[color_idx % len(mean_colors)], alpha=0.7,
                         linestyle="--")
"""
    # plot means
    for color_idx, mean in enumerate(bundle_avgs["mean"]):
        mean_regions, mean_idx_regions = region_split(mean, bundle_avgs["_idx"], float("nan"))
        for mean_region, mean_idx_region in zip(mean_regions, mean_idx_regions):
            ax_main.plot(mean_idx_region, mean_region, color=mean_colors[color_idx % len(mean_colors)], alpha=0.7)

    # plot legend
    patches = [mpatches.Patch(color=mean_colors[_i % len(mean_colors)],
                              label=_algo_label + " ({})".format(_mean_label))
               for _i, (_algo_label, _mean_label) in
               enumerate(zip(bundle_avgs["algo_label"], bundle_avgs["mean_label"]))]
    ax_main.legend(handles=patches)
    # plot running processes
    """
    endT_density_lst = [[len([_ for _x in run_endT_item if _x >= _t]) for _t in bundle_avgs["_idx"]] for run_endT_item
                        in bundle_avgs["endT"]]
    ax_fail.stackplot(bundle_avgs["_idx"],
                      endT_density_lst,
                      colors=[mean_colors[_i] for _i in range(len(bundle_avgs["mean"]))])

    # highlight process failure
    for _run_idx, failT_lst in enumerate(bundle_avgs["run_failT"]):
        for _t, failT in enumerate(failT_lst):
            if str(failT) != str(float("nan")):
                try:
                    ax_fail.text(failT, _run_idx, "{}!".format(_run_idx))
                except Exception as e:
                    pass
    """
    return fig

def show_config_diffs(mongo_central, bundles):
    # shows differences in config among several differences tags / runs
    config_hash = {}
    for bundle in bundles:
        for _instance_tag, _run_lst in bundle.items():  # iterate over all instance_tags
            for _run_id, _run_name in enumerate(_run_lst):  # iterate over all runs associate to _instance_tag
                #tag_names = mongo_central.get_tag_names(tag)
                config_res = mongo_central.get_name_prop(_run_name, "config")[0]["config"]
                flat_config_res = {}

                # flatten config_res
                for k, v in config_res.items():
                    if isinstance(v, dict):
                        for _k, _v in v.items():
                            flat_config_res[k+"."+_k] = _v
                    else:
                        flat_config_res[k] = v

                for k, v in flat_config_res.items():
                    if k in config_hash:
                        if v in config_hash[k]:
                            config_hash[k][v].append(_run_name)
                        else:
                            config_hash[k][v] = [_run_name]
                    else:
                        if k == "env_args": # this is itself a dict
                            for _k,_v in v.items():
                                config_hash[k] = {v: [_run_name]}
                        else:
                            key = v
                        config_hash[k] = {v:[_run_name]}

    print("Starting analysis...")
    import pprint
    pprint.pprint(config_hash)
    for k, v in config_hash.items():
        if len(v.keys()) > 1:
            print(v)
    pass

def main():
    root_dir = "/home/cs/Documents/pymarl-cqmix/src"
    mongo_central = MongoCentral(conf_names=["gandalf_pymarl"],
                                 root_dir=root_dir)
    # now start your experiments

    m1 = mongo_central.get_tag_names("F4I_FL_8m")
    pprint.pprint(m1)
    m2 = mongo_central.get_tag_names("F4I_FL_2_8m")
    pprint.pprint(m2)
    bavg = bundle_average(mongo_central,
                          [("MADDPG", m1, "Win rate test", "T env test"),
                           ("MADDPG", m2, "Win rate test", "T env test")],
                          no_fails=True,
                          t_res=40000)
    fig = plot_bundle_avgs(bavg)
    fig.savefig("8m")
    fig.show()

    # m1 = mongo_central.get_tag_names("F4I_FL_8m")
    # m2 = mongo_central.get_tag_names("F4I_FL_2_8m")
    # show_config_diffs(mongo_central, [m1, m2])

    pass




if __name__ == "__main__":
    main()