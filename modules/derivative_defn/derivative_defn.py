import numpy as np
import plotly.graph_objects as go
import os

def secant_line(f, a, h):
  '''
  Return the function of a secant line to the graph of a function at two points.

  Given a function f from R to R, compute an affine function l from R to R that passes through
  the points (a, f(a)) and (a + h, f(a + h)),

  Args:
    f (callable(x)): The function for which we want to compute the secant line. Input and output
      must be a scalar.
    a (float): The x-coordinate for one of the points on the secant line.
    h (float): The distance from a of the x-coordinate of the other point on the secant line.
  
  Returns:
    l (callable(x)): The function for the secant line. Input and output are scalars.
    m (float): The slope of the secant line.
  '''
  m = (f(a + h) - f(a)) / h
  def l(x): # y = m(x - x0) + y0
    return m * (x - a) + f(a)
  return l, m

def tangent_line(f, a, delta = 1e-6):
  '''
  Compute the function of a tangent line fo the graph of a fuction at a point.

  Given a function f from R to R, compute an affine function l from R to R that is tangent to 
  the graph of f at the point a. Approximate the slope m with a central finite difference with step
  size delta.

  Args:
    f (callable(x)): The function for which we want to compute the tangent line. Input and output
      must be a scalar.
    a (float): The x-coordinate of where we want to compute the tangent line.
    delta (float): The step size for central finite difference approximation of f'(a) (default
      is 1e-6)

  Returns:
    l (callable(x)): The function for the tangent line. Input and output are scalars.
    m (float): The slope of the tangent line.
  '''
  m = (f(a + delta) - f(a - delta)) / (2 * delta)
  def l(x):
    return m * (x - a) + f(a)
  return l, m

def make_figure(f, a, x_min, x_max, 
  save_figure = False, filename = "derivative_defn.html",
  square_aspect = True, tangent_line_always_visible = True,
  num_h = 100, num_x = 100,
  x_margin_ratio = 0.1, y_margin_ratio = 0.1,
  x_dtick = 0.5, y_dtick = 0.5,
  fx_width = 2, fx_color = "blue",
  a_size = 10, a_color = "black",
  tangent_line_width = 2, tangent_line_color = "deepskyblue",
  ah_size = 10, ah_color = "fuchsia",
  secant_line_width = 2, secant_line_color = "red",
  axis_width = 2, axis_color = "lightslategray"
  ):
  '''
  Draw the figure!

  Given a function f, a point a to approximate the derivative, and a range [x_min, x_max] for
    plotting things, generate a plotly plot with an interactive slider. Color-related arguments
    can be supplied as either a hex string (e.g. "#ffffff") or as a CSS named color (e.g. "red"). 
    
    See https://developer.mozilla.org/en-US/docs/Web/CSS/named-color for a list of CSS named colors.

  Args:
    f (callable(x)): The function for which we want to compute the secant line. Input and output
      must be a scalar.
    a (float): The x-coordinate of where we want to compute the tangent line.
    x_min (float): The left boundary of the domain where we want to compute things.
    x_max (float): The right boundary of the domain where we want to compute things.
    save_figure (boolean): Saves figure as html if True, displays inline if false (default is 
      False).
    filename (string): Filename for saved plot if save_figure == True (default is 
      derivative_defn.html)
    square_aspect (boolean): Whether or not to enforce a square aspect ratio for the plot; 
      recommended set to False if function over [x_min, x_max] has a particularly tall aspect ratio.
    tangent_line_always_visible (boolean): Tangent line always on if True, otherwise toggled off by
      default (default is True).
    num_h (int): Number of hs to compute secant value for (default is 100).
    num_x (int): Number of xs to compute f for (default is 100).
    x_margin_ratio (float): Multiple of x_max - x_min to pad on the left or right of the plot of
      x_max - x_min > y_max - y_min (default is 0.1).
    y_margin_ratio (float): Multiple of y_max - y_min to pad on the bottom or top of the plot of
      y_max - y_min > x_max - x_min (default is 0.1).
    x_dtick (float): Spacing between tick marks on x-axis (default is 0.5).
    y_dtick (float): Spacing between tick marks on y-axis (default is 0.5).
    a_size (float): Size of the point (a, f(a)) on the plot (default is 10).
    a_color (string): Color of the point (a, f(a)) (default is "black").
    fx_width (float): Width of the plot of y = f(x) (default is 2).
    fx_color (string): Color of the graph of y = f(x) (default is "blue").
    tangent_line_width (int): Thickness of the tangent line on the plot (default is 2).
    tangent_line_color (string): The color of the tangent line on the plot (default is 
      "deepskyblue").
    ah_size (float): Size of the point (a + h, f(a + h)) on the plot (default is 10).
    ah_color (string): Color of the point (a + h, f(a + h)) (default is "fuchsia").
    secant_line_width (int): Thickness of the tangent line on the plot (default is 2).
    secant_line_color (string): The color of the tangent line on the plot (default is "red").
    axis_width (float): The width of the cartesian axes (default is 2).
    axis_color (string): The color of the cartesian axes (default is "lightslategray").

  Returns: 
    None. Plots or saves figure.
  '''
  # check that a is in (x_min, x_max)
  if a <= x_min or a >= x_max:
    raise Exception(f"Value a must lie in range (x_min, x_max), got a = {a}, (x_min, x_max) = ({x_min}, {x_max}).")

  # generate x and y values for the function
  x = np.linspace(x_min, x_max, num_x)
  y = f(x)

  # set tangent line visibility flag
  if tangent_line_always_visible:
    tangent_line_visibility = True
  else:
    tangent_line_visibility = "legendonly"

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

  # plot the point (a, f(a))
  fig.add_trace(go.Scatter(x = [a], y = [f(a)], mode = "markers", 
    name = f"(a, f(a))", 
    marker = {'color' : a_color, 'size' : a_size},
    legendrank = 1))

  # plot the graph of y = f(x)
  fig.add_trace(go.Scatter(x = x, y = f(x), mode = 'lines', name = "f(x)", 
    line = {"color" : fx_color, "width" : fx_width},
    legendrank = 3))

  # plot the tangent line
  l, m = tangent_line(f, a)
  y0 = l(x_min)
  y1 = l(x_max)
  fig.add_trace(go.Scatter(x = [x_min, x_max], y = [y0, y1], mode = "lines", name = f"tangent line, slope = {m:.3f}",
    line = {"color" : tangent_line_color, "width" : tangent_line_width},
    visible = tangent_line_visibility, legendrank = 5))

  # compute hs
  h_min = x_min - a
  h_max = x_max - a
  hs = np.linspace(h_min, h_max, num_h)

  slopes = []  
  for h in hs:
    # plot the secant lines
    l, m = secant_line(f, a, h)
    y0 = l(x_min)
    y1 = l(x_max)
    slopes.append(m)
    fig.add_trace(
      go.Scatter(
        visible = False,
        x = [x_min, x_max],
        y = [y0, y1],
        mode = "lines",
        name = f"secant line",
        line = {"color" : secant_line_color, "width" : secant_line_width},
        legendrank = 4
      ))
    # plot the points (a + h, f(a + h))
    fig.add_trace(
      go.Scatter(
        visible = False,
        x = [a + h],
        y = [f(a + h)],
        mode = "markers", 
        name = "(a + h, f(a + h))", 
        marker = {"color" : ah_color, "size" : ah_size},
        legendrank = 2
      )
    )
  fig.data[3].visible = True
  fig.data[4].visible = True
  fig.update_layout(
    title = f"Slope = {slopes[0]:.3f}"
  )

  # configure sliders
  steps = []
  for i in range(num_h):
    step = dict(
      method="update",
      # set f(x) and (a, f(a)) to be permanently visible, tangent line to be visible in legend only
      args=[{"visible": [True, True, tangent_line_visibility] + [False] * 2 * num_h},
                    {"title": f"Slope = {slopes[i]:.3f}"}])
    step["args"][0]["visible"][2 * i + 3] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][2 * i + 4] = True
    steps.append(step)

  sliders = [dict(
      active=0,
      currentvalue={"prefix": "h =  "},
      pad={"t": 50},
      steps=steps
  )]

  fig.update_layout(
      sliders=sliders,
      xaxis_title="x",
      yaxis_title="y",
  )

  for i in range(num_h):
    fig['layout']['sliders'][0]['steps'][i]['label']=f"{hs[i]:.3f}"

  # plot x and y axes
  fig.update_xaxes(zeroline = True, zerolinewidth = axis_width, zerolinecolor = axis_color)
  fig.update_yaxes(zeroline = True, zerolinewidth = axis_width, zerolinecolor = axis_color)

  # save figure or display
  if save_figure:
    cwd = os.getcwd()
    print(cwd)
    print("figure saved to " + os.path.join(cwd, filename))
    fig.write_html(filename)

  else:
    fig.show()
      
    