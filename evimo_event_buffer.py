import numpy as np
import os
import bisect

def load_events(folder):
    events_t  = np.load(os.path.join(folder, 'dataset_events_t.npy'), mmap_mode='r')
    events_xy = np.load(os.path.join(folder, 'dataset_events_xy.npy'), mmap_mode='r')
    events_p  = np.load(os.path.join(folder, 'dataset_events_p.npy'), mmap_mode='r')

    events_t = np.atleast_2d(events_t.astype(np.float32)).transpose()
    events_p = np.atleast_2d(events_p.astype(np.float32)).transpose()

    events = np.hstack((events_t,
                        events_xy.astype(np.float32),
                        events_p))
    return events


class EvimoEventBuffer:
    def __init__(self, ev_dir) -> None:
        """
        ev_dir (str): samsung_mono/**/scene_name
        """
        tmp = load_events(ev_dir)
        self.t_f, self.x_f, self.y_f, self.p_f = [tmp[:,i] for i in range(4)]
        self.x_f, self.y_f = self.x_f.astype(np.uint16), self.y_f.astype(np.uint16)

        self.fs = [self.x_f, self.y_f, self.p_f, self.t_f]

        self.n_retrieve = 5000000

        self._init_cache()
        
    
    def _init_cache(self, idx=0):
        self.x_cache = np.array([self.x_f[idx]])
        self.y_cache = np.array([self.y_f[idx]])
        self.t_cache = np.array([self.t_f[idx]])
        self.p_cache = np.array([self.p_f[idx]])

        self.caches = [self.x_cache, self.y_cache, self.t_cache, self.p_cache]

        self.curr_pnter = idx + 1

    
    def update_cache(self):
        
        rx, ry, rp, rt = [e[self.curr_pnter:self.curr_pnter + self.n_retrieve] for e in self.fs]
        self.x_cache = np.concatenate([self.x_cache, rx])
        self.y_cache = np.concatenate([self.y_cache, ry])
        self.p_cache = np.concatenate([self.p_cache, rp])
        self.t_cache = np.concatenate([self.t_cache, rt])
        
        self.curr_pnter = min(len(self.t_f), self.curr_pnter + self.n_retrieve)

    def drop_cache_by_cond(self, cond):
        self.x_cache = self.x_cache[cond]
        self.y_cache = self.y_cache[cond]
        self.p_cache = self.p_cache[cond]
        self.t_cache = self.t_cache[cond]
    
    def clear_cache(self):
        self.x_cache = np.array([])
        self.y_cache = np.array([])
        self.t_cache = np.array([])
        self.p_cache = np.array([])
    

    def flip_polarity(self, polarity):
        ptype = polarity.dtype
        return (~polarity.astype(bool)).astype(ptype)

    def retrieve_data(self, st_t, end_t, is_far=False):
        # NOTE: THIS FUNCTION FLIP POLARITY SINCE SAMSUNG CAMERA IS OPPOSITE
        if (len(self.t_cache) == 0) or (self.t_cache[0] > st_t) or is_far:
            ## if st_t already out of range
            idx = bisect.bisect(self.t_f, st_t)
            idx = idx if ((st_t == self.t_f[idx]) or st_t <= self.t_f[0]) else idx - 1

            assert idx >= 0, f"{st_t} not found!!"

            self._init_cache(idx)


        while (self.curr_pnter < len(self.t_f)) and (self.t_cache[-1] <= end_t):
            self.update_cache()
        
        ret_cond = ( st_t<= self.t_cache) & (self.t_cache <= end_t)
        ret_data = [self.t_cache[ret_cond], self.x_cache[ret_cond], 
                    self.y_cache[ret_cond], self.flip_polarity(self.p_cache[ret_cond])]
        self.drop_cache_by_cond(~ret_cond)

        return ret_data
    
    def drop_cache_by_t(self, t):
        cond = self.t_cache >= t
        self.drop_cache_by_cond(cond)
    
    def validate_time(self, st_t):
        return st_t < self.t_f[-1]

    def drop_cache_by_t(self, t):
        cond = self.t_cache >= t
        self.drop_cache_by_cond(cond)