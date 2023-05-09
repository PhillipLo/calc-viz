import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os


def compute_surface(f, x_min, x_max, y_min, y_max, num_x = 100, num_y = 100):
  xs = np.linspace(x_min, x_max, num_x)
  ys = np.linspace(y_min, y_max, num_y)
  xx, yy = np.meshgrid(xs, ys)

  grid = np.stack((xx.T, yy.T), axis = 2)
  grid_flat = np.reshape(grid, [num_x * num_y, 2])

  zz = np.reshape(f(grid_flat[:, 0], grid_flat[:, 1]), (num_x, num_y))

  return xx, yy, zz

def get_contours(xx, yy, zz, cs):
  cn = plt.contour(xx, yy, zz, cs)
  plt.close()
  contours = []
  # for each contour line
  for cc in cn.collections:
    paths = []
    # for each separate section of the contour line
    for pp in cc.get_paths():
      xy = []
      # for each segment of that section
      for vv in pp.iter_segments():
        xy.append(vv[0])
      paths.append(np.vstack(xy))
    contours.append(paths)
  

  return contours

def make_figure(f, x_min, x_max, y_min, y_max, c_min, c_max, 
  num_x = 100, num_y = 100, num_c = 10,
  plane_opacity = 0.9, surface_opacity = 0.5,
  level_curve_width = 3, level_curve_color = "black",
  intersection_width = 3, intersection_color = "green",
  filename = "level_curves.html", save_figure = False,
  title = "Level Curves of f(x,y)"):


  xx, yy, zz = compute_surface(f, x_min, x_max, y_min, y_max, num_x = num_x, num_y = num_y)
  cs = np.linspace(c_min, c_max, num_c)
  
  contours = get_contours(xx, yy, zz, cs)

  fig = go.Figure()
  func_surface = go.Surface(x = xx, y = yy, z = zz, opacity = surface_opacity, hoverinfo = "skip", showlegend = False, showscale = False)  

  fig.add_trace(func_surface)
  
  num_traces_per_contour = []

  for i in range(num_c):
    num_components = len(contours[i]) # number of connected components of contour
    num_traces_per_contour.append(num_components * 2 + 1)
    plane = go.Surface(x = xx, y = yy, z = cs[i] * np.ones(shape = (num_x, num_y)), opacity = plane_opacity, visible = False, hoverinfo = "skip", showlegend = False, showscale = False)
    fig.add_trace(plane)
    for j in range(num_components):
      contour_len = contours[i][j].shape[0]
      
      level_curve = go.Scatter3d(x = contours[i][j][:, 0], y = contours[i][j][:, 1], z = np.zeros(shape = (contour_len,)), mode = 'lines', line = dict(width = level_curve_width, color = level_curve_color), visible = False, hoverinfo = "skip", showlegend = False)
      intersection = go.Scatter3d(x = contours[i][j][:, 0], y = contours[i][j][:, 1], z = cs[i] * np.ones(shape = (contour_len,)), mode = 'lines', line = dict(width = intersection_width, color = intersection_color), visible = False, hoverinfo = "skip", showlegend = False)
      

      fig.add_trace(level_curve)
      fig.add_trace(intersection)
      
  stop_idxs = 1 + np.cumsum(num_traces_per_contour)
  stop_idxs = np.insert(stop_idxs, 0, 1)
  for k in range(1, stop_idxs[1]):
    fig.data[k].visible = True

  fig.update_layout(title = f"Level Curve at c = {cs[0]:.3f}")
  fig.update_layout(coloraxis_showscale=False)
  # configure sliders
  steps = []
  for i in range(num_c):
    step = dict(
      method="update",
      # set f(x) and (a, f(a)) to be permanently visible, tangent line to be visible in legend only
      args=[{"visible": [False] * len(fig.data)},
                    {"title": f"Level Curve at c = {cs[i]:.3f}"}])
    step["args"][0]["visible"][0] = True  # Toggle i'th trace to "visible"

    for k in range(stop_idxs[i], stop_idxs[i + 1]):
      step["args"][0]["visible"][k] = True
    steps.append(step)

  sliders = [dict(
      active=0,
      currentvalue={"prefix": "c =  "},
      pad={"t": 50},
      steps=steps
  )]

  fig.update_layout(
      sliders=sliders,
      xaxis_title="x",
      yaxis_title="y",
  )

  for i in range(num_c):
    fig['layout']['sliders'][0]['steps'][i]['label']=f"{cs[i]:.3f}"
  
  if save_figure:
    cwd = os.getcwd()
    print(cwd)
    print("figure saved to " + os.path.join(cwd, filename))
    fig.write_html(filename)

  else:
    fig.show()


def plot_contours(f, x_min, x_max, y_min, y_max, c_min, c_max, 
  num_x = 100, num_y = 100, num_c = 10):

  xx, yy, zz = compute_surface(f, x_min, x_max, y_min, y_max, num_x = num_x, num_y = num_y)
  cs = np.linspace(c_min, c_max, num_c)
  plt.xlim([x_min, x_max])
  plt.ylim([y_min, y_max])
  ax = plt.gca()
  ax.set_aspect('equal', adjustable='box')
  plt.contour(yy, xx, zz, cs)


