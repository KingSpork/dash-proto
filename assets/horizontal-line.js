//

// Initialize tool behavior for all graphs
document.addEventListener('DOMContentLoaded', function() {
  const graphs = document.querySelectorAll('.js-plotly-plot');
  graphs.forEach(function(graph) {
    handleHorizontalLineCreation(graph);
  });
});

function handleHorizontalLineCreation(gd) {
  // Store the original relayout function
  const originalRelayout = Plotly.relayout;

  // Custom handler for horizontal line creation
  gd.on('plotly_relayout', function(eventData) {
    // Only proceed if we're in horizontal line mode
    if (!gd._horizontalLineMode) return;

    // Check if a new shape was added (look for shape array changes)
    const shapeKeys = Object.keys(eventData).filter(k => k.startsWith('shapes['));
    if (shapeKeys.length === 0) return;

    // Get the last shape index
    const lastIndex = gd._fullLayout.shapes.length - 1;
    const newShape = gd._fullLayout.shapes[lastIndex];

    // Verify it's a line (our starting point)
    if (newShape && newShape.type === 'line') {
      // Convert to horizontal line
      const yValue = newShape.y0;

      // Use the original relayout to avoid infinite loops
      originalRelayout(gd, {
        [`shapes[${lastIndex}]`]: {
          x0: 0,
          x1: 1,
          xref: 'paper', // Makes it span the entire x-axis
          y0: yValue,
          y1: yValue, // Ensures perfect horizontality
          type: 'line',
          line: {
            color: newShape.line.color,
            width: newShape.line.width,
            dash: newShape.line.dash
          }
        }
      });

      // Reset the mode
      gd._horizontalLineMode = false;
    }
  });
}

function enforceHorizontalConstraints(gd) {
  gd.on('plotly_restyle', function(eventData) {
    // Get current shapes or exit if none exist
    const shapes = gd._fullLayout.shapes || [];
    if (shapes.length === 0) return;

    // Object to collect updates
    const updates = {};

    shapes.forEach((shape, index) => {
      // Skip if not one of our horizontal lines
      if (!(shape.xref === 'paper' && shape.y0 === shape.y1)) return;

      const shapePrefix = `shapes[${index}]`;

      // Case 1: Someone tried to change x-coordinates
      if (eventData[`${shapePrefix}.x0`] !== undefined ||
          eventData[`${shapePrefix}.x1`] !== undefined) {
        updates[`${shapePrefix}.x0`] = 0;
        updates[`${shapePrefix}.x1`] = 1;
      }

      // Case 2: Someone tried to change y1 (to make non-horizontal)
      if (eventData[`${shapePrefix}.y1`] !== undefined &&
          eventData[`${shapePrefix}.y1`] !== eventData[`${shapePrefix}.y0`]) {
        updates[`${shapePrefix}.y1`] = updates[`${shapePrefix}.y0`] || shape.y0;
      }

      // Case 3: Someone tried to change both y0 and y1 differently
      if (eventData[`${shapePrefix}.y0`] !== undefined &&
          eventData[`${shapePrefix}.y1`] !== undefined &&
          eventData[`${shapePrefix}.y0`] !== eventData[`${shapePrefix}.y1`]) {
        const newY = eventData[`${shapePrefix}.y0`];
        updates[`${shapePrefix}.y0`] = newY;
        updates[`${shapePrefix}.y1`] = newY;
      }
    });

    // Apply updates if needed
    if (Object.keys(updates).length > 0) {
      Plotly.relayout(gd, updates);
    }
  });
}

console.log('Horizontal line tool initialized');