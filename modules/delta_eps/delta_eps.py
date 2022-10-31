import numpy as np
import plotly.graph_objects as go

def compute_delta(f, a, L, eps, x_min, x_max, delta_stepsize = 1e-5):
  '''
  Given a function f where \lim_{x\to a}f(x) = L and an epsilon > 0, find the corresponding maximum
    delta such that if 0 < |x - a| < delta, then |f(x) - L| < epsilon.
  
  Computes delta by stepping to the left and right of a in steps of size delta_stepsize and taking
    the minumum of the leftmost and rightmost step from a.  

  Args:
    f (callable (x)): The function itself. Input and output must be a scalar.
    a (float): The x coordinate of where we want to compute the limit.
    L (float): The limit.
    eps (float > 0): The epsilon for which we want to find a delta.
    x_min (float): The left boundary of the domain where we want to compute things.
    x_max (float): The right boundary of the domain where we want to compute things.
    delta_stepsize (float): The stepsize we use for finding delta (default is 1e-5).

  Returns:
    delta (float): The computed delta.
  '''
  if eps <= 0: 
    raise Exception(f"Argument eps must be positive, got {eps} = eps.")
    
  delta_l = 0
  while a - delta_l > x_min + delta_stepsize:
    if f(a - delta_l) < L + eps and f(a - delta_l) > L - eps:
      delta_l += delta_stepsize
    else:
      break

  delta_r = 0
  while a + delta_r < x_max - delta_stepsize:
    if f(a + delta_r) < L + eps and f(a + delta_r) > L - eps:
      delta_r += delta_stepsize
    else:
      break

  delta = min(delta_r, delta_l)

  return delta

def make_figure(f, a, L, x_min, x_max,
  save_figure = False, filename = "delta_eps.html",
  num_x = 1000,
  eps_min = 0.05, eps_max = 0.5, num_eps = 20,
  delta_stepsize = 1e-5,
  a_size = 10, a_color = "black",
  fx_width = 2, fx_color = "blue",
  vline_width = 2, vline_color = "red",
  vbar_width = 1, vbar_color = "pink",
  hline_width = 2, hline_color = "orange",
  hbar_width = 1, hbar_color = "orangered",
  axis_width = 2, axis_color = "lightslategray",
  x_dtick = 0.5, y_dtick = 0.5,
  square_aspect = True,
  x_margin_ratio = 0.1, y_margin_ratio = 0.1):
  '''
  Draw the figure!

  Given a function f such that \lim_{x\to a}f(x) =  L, generate a plotly plot with an interactive
    slider for epsilon. Given an epsilon selected by the slider, draw the horizontal band spanning 
    (L - \epsilon, L + \epsilon) and the vertical band spanning (a - \delta, a + \delta), where 
    \delta is the maximum delta such that if 0 < |x - a| < \delta, then |f(x) - L| < \epsilon.

  Args:
    f (callable (x)): The function itself. Input and output must be a scalar.
    a (float): The x coordinate of where we want to compute the limit.
    L (float): The limit.
    save_figure (boolean): 
    eps (float > 0): The epsilon for which we want to find a delta.
    x_min (float): The left boundary of the domain where we want to compute things.
    x_max (float): The right boundary of the domain where we want to compute things.
    save_figure (boolean): Saves figure as html if True, displays inline if false (default is 
      False).
    filename (string): Filename for saved plot if save_figure == True (default is delta_eps.html).
    num_x (int): Number of xs to compute f for (default is 100).
    eps_min (float): Minimum range of epsilons to plot things for (default is 0.05).
    eps_max (float): Maximum range of epsilons to plot things for (default is 0.5).
    num_eps (int): Number of epsilons between eps_min and eps_max (inclusive) to plot things for
      (default is 20).
    delta_stepsize (float): The stepsize we use for finding delta (default is 1e-5).
    a_size (float): The size of the point (a, f(a)) (default is 10).
    a_color (string): The color of the point (a, f(a)) (default is "black").
    fx_width (float): The thickness of the plot of f(x) (default is 2).
    fx_color (string): The color of the plot of f(x) (default is "blue").
    vline_width (float): The thickness of the vertical line x = a (default is 2).
    vline_color (string): The color of the vertical line x = a (default is "red").
    vbar_width (float): The thickness of the borders of the vertical delta range band (default
      is 1).
    vbar_color (string): The color of the vertical delta range band (default is "pink").
    hline_width (float): The thickness of the horizontal line y = L (default is 2).
    hline_color (string): The color of the horizontal line y = L (default is "orange").
    hbar_width (float): The thickness of the borders of the horizontal epsilon range band (default
      is 1).
    hbar_color (string): The color of the horizontal delta range band (default is "orangered").
    axis_width (float): The width of the cartesian axes (default is 2).
    axis_color (string): The color of the cartesian axes (default is "lightslategray").
    x_dtick (float): Spacing between tick marks on x-axis (default is 0.5).
    y_dtick (float): Spacing between tick marks on y-axis (default is 0.5).
    square_aspect (boolean): Whether or not to enforce a square aspect ratio for the plot; 
      recommended set to False if function over [x_min, x_max] has a particularly tall aspect ratio.
    x_margin_ratio (float): Multiple of x_max - x_min to pad on the left or right of the plot of
      x_max - x_min > y_max - y_min (default is 0.1).
    y_margin_ratio (float): Multiple of y_max - y_min to pad on the bottom or top of the plot of
      y_max - y_min > x_max - x_min (default is 0.1).

  Returns: 
    None. Plots or saves figure.
  '''
  # check that a is in (x_min, x_max)
  if a <= x_min or a >= x_max:
    raise Exception(f"Value a must lie in range (x_min, x_max), got a = {a}, (x_min, x_max) = ({x_min}, {x_max}).")

  if eps_min >= eps_max:
    raise Exception(f"Value of eps_min must be less than eps_max, got eps_min = {eps_min}, eps_max = {eps_max}.")

  # generate x and y values for the function
  x = np.linspace(x_min, x_max, num_x)
  y = f(x)

  # configure plot shape
  y_min = np.min(y)
  y_max = np.max(y)

  x_margin = x_margin_ratio * (x_max - x_min)
  y_margin = y_margin_ratio * (y_max - y_min)

  fig = go.Figure()

  fig.update_xaxes(range=[x_min - x_margin, x_max + x_margin], tick0 = np.floor(x_min - x_margin), dtick = x_dtick)
  fig.update_yaxes(
    range=[y_min - y_margin, y_max + y_margin],
    autorange=False,
    tick0 = np.floor(y_min - y_margin),
    dtick = y_dtick
  )
  if square_aspect:
    fig.update_yaxes(
    scaleratio = 1,
    scaleanchor = "x",
    )

  # plot the point (a, L)
  fig.add_trace(go.Scatter(x = [a], y = [L], mode = "markers", 
    name = f"(a, f(a))", 
    marker = {'color' : a_color, 'size' : a_size},
    legendrank = 1))

  # plot the graph of y = f(x)
  fig.add_trace(go.Scatter(x = x, y = f(x), mode = 'lines', name = "f(x)", 
    line = {"color" : fx_color, "width" : fx_width},
    legendrank = 3))

  # plot the lines y = L and x = delta
  fig.add_trace(go.Scatter(visible = True, x = [x_min, x_max], y = [L, L], 
    mode = "lines", name = "y = L", line = {"color" : hline_color, "width" : hline_width}))
  fig.add_trace(go.Scatter(visible = True, x = [a, a], y = [y_min, y_max], 
    mode = "lines", name = "x = δ", line = {"color" : vline_color, "width" : vline_width}))

  # plot slider traces: 
  epsilons = np.linspace(eps_max, eps_min, num_eps)
  deltas = []
  for eps in epsilons:
    delta = compute_delta(f, a, L, eps, x_min, x_max, delta_stepsize)
    deltas.append(delta)
    # the horizontal band for epsilon
    fig.add_trace(go.Scatter(visible = False, x = [x_min, x_max, x_max, x_min, x_min], 
      y = [L - eps, L - eps, L + eps, L + eps, L - eps], 
      mode = "lines", fill = "toself", name = "(L - ɛ, L + ɛ)", line = {"color" : hbar_color, "width" : hbar_width}))
    # the vertical band for delta
    fig.add_trace(go.Scatter(visible = False, x = [a - delta, a + delta, a + delta, a - delta, a - delta], 
      y = [y_min, y_min, y_max, y_max, y_min], 
      mode = "lines", fill = "toself", name = "(a - δ, a + δ)", line = {"color" : vbar_color, "width" : vbar_width}))

  fig.data[4].visible = True
  fig.data[5].visible = True
  fig.update_layout(title = f"ɛ = {epsilons[0]:.2f}, δ = {deltas[0]:.2f}")

  steps = []
  # display epsilon and delta in figure title
  for i in range(num_eps):
    step = dict(
      method="update",
      args = [{"visible": [True, True, True, True] + [False] * 2 * num_eps}, 
      {"title" : f"ɛ = {epsilons[i]:.2f}, δ = {deltas[i]:.2f}"}]
    )
    for j in range(4 + i * 2, 4 + i * 2 + 2):
      step["args"][0]["visible"][j] = True
    steps.append(step)

  # format slider labels
  sliders = [dict(
    active=0,
    currentvalue={"prefix": "ɛ =  "},
    pad={"t": 50},
    steps=steps
    )]

  fig.update_layout(
      sliders=sliders,
      xaxis_title="x",
      yaxis_title="y",
  )

  for i in range(num_eps):
    fig['layout']['sliders'][0]['steps'][i]['label']=f"{epsilons[i]:.2f}"
  
  # plot x and y axes
  fig.update_xaxes(zeroline = True, zerolinewidth = axis_width, zerolinecolor = axis_color)
  fig.update_yaxes(zeroline = True, zerolinewidth = axis_width, zerolinecolor = axis_color)

  # save figure or display
  if save_figure:
    fig.write_html(filename)
  else:
    fig.show()







