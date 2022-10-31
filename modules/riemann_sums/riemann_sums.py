import numpy as np
import plotly.graph_objects as go
import os

def get_corners_of_ith_rectangle(f, a, b, i, N, mode):
  '''
  Compute the coordinates of the corners needed to plot a rectangle as part of a Riemann sum.

  Given a function f from [a, b] to R, the Riemann sum with N subintervals can be plotted as N 
    rectangles. This function computes the coordinates of the corners of the ith rectangle.

  Args:
    f (callable(x)): The function to integrate. Input and output must be a scalar.
    a (float) : The left boundary of the domain of integration.
    b (float) : The right boundary of the domain of integration.
    i (int) : The index of the rectangle we with to compute the coordinates for; the leftmost 
      rectangle has index 1, the rightmost has index N; i must be between 1 and N (inclusive).
    N (int): The total number of subdivisions for the Riemann sum.
    mode (string): Either "left", "right", or "midpoint"; specifies the type of Riemann sum.

  Returns:
    x (list): List of length 5; the x coordinates of the corners of the rectangle. There are 5 
      rather than 4 coordinates because the first point needs to be repeated to close the rectangle.
    y (list): List of length 5; the y coordinates of the corners of the rectangle. There are 5 
      rather than 4 coordinates because the first point needs to be repeated to close the rectangle.
    area (float): The signed area of the rectangle.

  '''
  xl = a + (i - 1) * (b - a) / N
  xr = a + i * (b - a) / N 
  
  if mode == "left":
    h = f(xl)
  elif mode == "right":
    h = f(xr)
  elif mode == "midpoint":
    xm = (xr + xl) / 2
    h = f(xm)

  x = [xl, xr, xr, xl, xl]
  y = [0, 0, h, h, 0]

  area = h * (xr - xl)
  return x, y, area


def compute_trace_idxs(max_N):
  '''
  Compute indices needed to display the correct traces.

  The zeroth trace in the final plotly figure is the graph of the function itself. The first trace
    is the single rectangle from the Riemann approximation with N = 1. Traces 2 - 3 are the two
    rectangles from the Riemann approximation with N = 2, etc. This function helps with computing
    these indices.

  Args:
    max_N (int): The maximum number of partitions to compute Riemann sums for.
  
  Returns:
    trace_idxs (list): List of length max_N + 1; indices for plotly traces.
  '''
  trace_idxs = [0]
  for i in range(1, max_N + 1):
    trace_idxs.append(trace_idxs[-1] + i)
  return trace_idxs


def make_figure(f, a, b, mode,
  save_figure = False, filename = "riemann_sums.html",
  num_x = 100, max_N = 30,
  square_aspect = True,
  x_margin_ratio = 0.1, y_margin_ratio = 0.1,
  x_dtick = 0.5, y_dtick = 0.5,
  fx_width = 2, fx_color = "black",
  rect_line_width = 1, rect_pos_color = "blue", rect_neg_color = "red",
  axis_width = 2, axis_color = "lightslategray"
  ):
  '''
  Draw the figure!

  Given a function f, a domain [a, b], and a Riemann sum mode "left", "right", or "midpoint", plot
    the function and the rectangles representing the Riemann sum, with the number of rectangles N
    configurable by a slider. Color-related arguments can be supplied as either a hex string 
    (e.g. "#ffffff") or as a CSS named color (e.g. "red"). 
    
    See https://developer.mozilla.org/en-US/docs/Web/CSS/named-color for a list of CSS named colors.

  Args:
    f (callable(x)): The function to integrate. Input and output must be a scalar.
    a (float) : The left boundary of the domain of integration.
    b (float) : The right boundary of the domain of integration.
    mode (string): Either "left", "right", or "midpoint"; specifies the type of Riemann sum.
    save_figure (boolean): Saves figure as html if True, displays inline if false (default is 
      False).
    filename (string): Filename for saved plot if save_figure == True (default is riemann_sums.html)
    square_aspect (boolean): Whether or not to enforce a square aspect ratio for the plot; 
      recommended set to False if function over [x_min, x_max] has a particularly tall aspect ratio.
    x_margin_ratio (float): Multiple of x_max - x_min to pad on the left or right of the plot of
      x_max - x_min > y_max - y_min (default is 0.1).
    y_margin_ratio (float): Multiple of y_max - y_min to pad on the bottom or top of the plot of
      y_max - y_min > x_max - x_min (default is 0.1).
    x_dtick (float): Spacing between tick marks on x-axis (default is 0.5).
    y_dtick (float): Spacing between tick marks on y-axis (default is 0.5).
    fx_width (float): Width of the plot of y = f(x) (default is 2).
    fx_color (string): Color of the graph of y = f(x) (default is "black").
    rect_line_width (float): Width of the borders of the rectangles (default is 1).
    rect_pos_color (string): Color of rectangles with positive signed area (default is "blue").
    rect_neg_color (string): Color of rectangles with negative signed area (default is "red").
    axis_width (float): The width of the cartesian axes (default is 2).
    axis_color (string): The color of the cartesian axes (default is "lightslategray").
  '''
  # check that a < b
  if a >= b:
    raise Exception(f"The value of a must be strictly less than b, got a = {a}, b = {b}.")
  
  # generate x and y values for the function
  x = np.linspace(a, b, num_x)
  y = f(x)

  # Ns = [1, 2, ..., max_N]
  Ns = np.arange(1, max_N + 1)

  # configure plot shape
  y_min = np.min(y)
  y_max = np.max(y)

  x_margin = x_margin_ratio * (b - a)
  y_margin = y_margin_ratio * (y_max - y_min)

  fig = go.Figure()

  fig.update_xaxes(range=[a - x_margin, b + x_margin], tick0 = np.floor(a - x_margin), dtick = x_dtick)
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
  
  # plot the graph of y = f(x)
  fig.add_trace(go.Scatter(x = x, y = f(x), mode = 'lines', name = "f(x)", 
    line = {"color" : fx_color, "width" : fx_width},
    legendrank = 1))

  # plot the rectangles and compute the Riemann sums
  areas = []
  for N in Ns:
    area_sum = 0
    for i in range(1, N + 1):
      rect_i_x, rect_i_y, area = get_corners_of_ith_rectangle(f, a, b, i, N, mode)
      area_sum += area
      if area >= 0:
        rect_color = rect_pos_color
      else:
        rect_color = rect_neg_color
      fig.add_trace(go.Scatter(visible = False, showlegend = False, 
        x = rect_i_x, y = rect_i_y, 
        fill = "toself", mode = "lines", 
        line = {"color" : rect_color, "width" : rect_line_width},
        name = f"A_{i}"))
    areas.append(area_sum)

  fig.data[1].visible = True
  fig.update_layout(title = mode + 
    f" Riemann sum for f with N = 1, Δx = {(b - a)/N:.3f}, area = {areas[0]:.3f}")

  # configure sliders
  trace_idxs = compute_trace_idxs(max_N)

  steps = []
  for N in Ns:
    step = dict(
      method="update",
      args = [{"visible": [True] + [False] * trace_idxs[-1]}, 
      {"title" : mode + 
        f" Riemann sum for f with N = {N}, Δx = {(b - a)/N:.3f}, area = {areas[N - 1]:.3f}"}]
    )
    for j in range(trace_idxs[N - 1], trace_idxs[N]):
      step["args"][0]["visible"][j + 1] = True
    steps.append(step)

  sliders = [dict(
    active=0,
    currentvalue={"prefix": "N =  "},
    pad={"t": 50},
    steps=steps
    )]

  fig.update_layout(
      sliders=sliders,
      xaxis_title="x",
      yaxis_title="y",
  )

  for i in range(max_N):
    fig['layout']['sliders'][0]['steps'][i]['label']=f"{Ns[i]}"

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




  


