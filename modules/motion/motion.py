import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.misc import derivative
import os
import warnings

def compute_psn_vel_acc(p, t_min, t_max, num_ts = 10000):
  '''
  Given a scalar function p(t) and a collection of ts to evaluate s at, compute the position,
    velocity, and acceleration at each of the times t in ts.

  Consider a particle in motion whose position at time t is given by p(t). Let v(t) and a(t)
    denote the velocity and acceleration, respectively. We compute s, v, and a (psns, vels, and 
    accs) using forward finite differences.

  Args:
    p (callable(x)): The position function. Input and output must be a scalar.
    t_min (float): Left boundary of the domain where we want to compute s(t).
    t_max (float): Right boundary of the domain where we want to compute s(t). Recommended that
      t_max > t_min + 0.01 for numerical reasons.
    num_ts (int): Number of time points to compute s(t) at (default is 10000). Recommended range
      [5000, 10000] for numerical reasons.
      
  Returns:
    ts: ((num_ts,)-shaped array): The times where we compute things.
    psns: ((num_ts,)-shaped array): The positions of the particle at times ts.
    vels: ((num_ts,)-shaped array): The velocities of the particle at times ts.
    accs: ((num_ts,)-shaped array): The accelerations of the particle at times ts.
  '''
  if t_max <= t_min:
    raise Exception(f"t_max must be strictly greater than t_min, got t_min = {t_min}, t_max = {t_max}.")

  if t_max <= t_min + 0.01:
    warnings.warn(f"Recommended that t_max > t_min + 0.01 for numerical reasons; got t_min = {t_min}, t_max = {t_max}", RuntimeWarning)

  if num_ts < 5000 or num_ts > 10000:
    warnings.warn(f"Recommended that num_ts in range [5000, 10000] for numerical reasons, got num_ts = {num_ts}", RuntimeWarning)

  ts = np.linspace(t_min, t_max, num_ts)
  h = ts[1] - ts[0]

  psns = p(ts)
  vels = derivative(p, ts, dx = h)
  accs = derivative(p, ts, dx = h, n = 2)

  return ts, psns, vels, accs


def make_figure(p, t_min, t_max,
  num_ts = 10000, downsampling_ratio = 200,
  ):
  '''
  Draw the figure!

  Given a function p(t) describing the position of a particle in 1D motion, simultaneously animate
    the particle itself along with the position, velocity, and acceleration of the particle plotted
    against time. The animation is controllable with start/stop buttons and a slider.

  Args:
    p (callable(x)): The position function. Input and output must be a scalar.
    t_min (float): Left boundary of the domain where we want to compute s(t).
    t_max (float): Right boundary of the domain where we want to compute s(t). Recommended that
      t_max > t_min + 0.01 for numerical reasons.
    num_ts (int): Number of time points to compute s(t) at (default is 10000). Recommended range
      [5000, 10000] for numerical reasons.
    downsampling_ratio (int): Downsampling ratio for frames; number of frames in animation is 
      num_ts // downsampling_ratio; recommended to choose such that number of frames is between
      100 and 1000. Increasing downsampling_ratio will result in a faster animation, and vice
      versa (default value is 200)
  '''
  # get data
  ts, psns, vels, accs = compute_psn_vel_acc(p, t_min, t_max, num_ts = num_ts)

  # initialize plotly figure with subplots
  fig = go.Figure().set_subplots(4,1, row_heights = [0.1, 0.3, 0.3, 0.3],
    specs=[[{"type": "scatter"}], [{"type":"scatter"}], [{"type":"scatter"}], [{"type":"scatter"}]])


  margin_ratio = 0.1

  ts_margin = margin_ratio * (np.max(ts) - np.min(ts))
  psns_margin = margin_ratio * (np.max(psns) - np.min(psns))
  vels_margin = margin_ratio * (np.max(vels) - np.min(vels))
  accs_margin = margin_ratio * (np.max(accs) - np.min(accs))

  fig.update_layout(
    {    
        'xaxis1':{'range':[np.min(psns) - psns_margin, np.max(psns) + psns_margin], "title":"position"},
        'yaxis1':{'range':[-1, 1], "visible":False, "zeroline":True},
        'xaxis2':{'range':[np.min(ts) - ts_margin, np.max(ts) + ts_margin], "title":"t"},
        'yaxis2':{'range':[np.min(psns) - psns_margin, np.max(psns) + psns_margin], "title":"p"},
        'xaxis3':{'range':[np.min(ts) - ts_margin, np.max(ts) + ts_margin], "title":"t"},
        'yaxis3':{'range':[np.min(vels) - vels_margin, np.max(vels) + vels_margin], "title":"v"},
        'xaxis4':{'range':[np.min(ts) - ts_margin, np.max(ts) + ts_margin], "title":"t"},
        'yaxis4':{'range':[np.min(accs) - accs_margin, np.max(accs) + accs_margin], "title":"a"}
    })

  fig.add_trace(go.Scatter(x = [psns[0]], y= [0], mode = "markers", hoverinfo = "skip"), 1, 1)
  fig.add_trace(go.Scatter(x = ts[0:1], y = psns[0:1], mode = "lines", hoverinfo = "skip"), 2, 1)
  fig.add_trace(go.Scatter(x = ts[0:1], y = vels[0:1], mode = "lines", hoverinfo = "skip"), 3, 1)
  fig.add_trace(go.Scatter(x = ts[0:1], y = accs[0:1], mode = "lines", hoverinfo = "skip"), 4, 1)

  frames = [ 
    go.Frame(
      data = [
        go.Scatter(x = [psns[i]], y = [0], mode = "markers", hoverinfo = "skip"),
        go.Scatter(x = ts[0:i+1], y = psns[0:i+1], mode = "lines", hoverinfo = "skip"),
        go.Scatter(x = ts[0:i+1], y = vels[0:i+1], mode = "lines", hoverinfo = "skip"),
        go.Scatter(x = ts[0:i+1], y = accs[0:i+1], mode = "lines", hoverinfo = "skip")
      ],
      name = f"t = {ts[i]}",
      traces = [0, 1, 2, 3]) for i in range(1, num_ts, downsampling_ratio)
  ]

  fig.update(frames = frames)

  def frame_args(duration):
      return {
              "frame": {"duration": duration},
              "mode": "immediate",
              "fromcurrent": True,
              #"transition": {"duration": duration, "easing": "linear"},
          }

  fr_duration=1  # customize this frame duration according to your data!!!!!
  sliders = [
              {
                  "pad": {"b": 10, "t": 50},
                  "len": 0.9,
                  "x": 0.1,
                  "y": 0,
                  "steps": [
                      {
                          "args": [[f.name], frame_args(fr_duration)],
                          "label": f"fr{k+1}",
                          "method": "animate",
                      }
                      for k, f in enumerate(fig.frames)
                  ],
              }
          ]


  fig.update_layout(sliders=sliders,
                    updatemenus = [
                          {
                          "buttons": [
                              {
                              "args": [None, frame_args(fr_duration)],
                              "label": "&#9654;", # play symbol
                              "method": "animate",
                              },
                              {
                              "args": [[None], frame_args(fr_duration)],
                              "label": "&#9724;", # pause symbol
                              "method": "animate",
                              }],
                          "direction": "left",
                          "pad": {"r": 10, "t": 70},
                          "type": "buttons",
                          "x": 0.1,
                          "y": 0,
                          }])

  fig.write_html("anim.html")
    

