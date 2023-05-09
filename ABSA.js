fetch('http://127.0.0.1:5000/aspects')
  .then(response => response.json())
  .then(data => {
    // You can access the JSON data in the 'data' variable
    console.log(data);
     // Update the content of a <div> element with the JSON data
    //const divElement = document.getElementById('dataContainer');
    //divElement.textContent = data;
    const newData = JSON.parse(data);

    const container = d3.select('#dataContainer');

    const width = 400;
    const height = 300;
    const margin = { top: 20, right: 50, bottom: 30, left: 100 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const svg = container.append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    const barData = Object.entries(newData).map(([key, value]) => ({ key, value }));
    const xScale = d3.scaleLinear()
        .domain([0, d3.max(barData, d => d.value)])
        .range([0, chartWidth]);

    const yScale = d3.scaleBand()
        .domain(barData.map(d => d.key))
        .range([0, chartHeight])
        .padding(0.1);
    const bars = svg.selectAll(".bar")
        .data(barData)
        .enter()
        .append("rect")
        .attr("class", "bar")
        .attr("y", d => yScale(d.key))
        .attr("width", d => xScale(d.value))
        .attr("height", yScale.bandwidth());
    const labels = svg.selectAll(".label")
        .data(barData)
        .enter()
        .append("text")
        .attr("class", "label")
        .attr("x", d => xScale(d.value))
        .attr("y", d => yScale(d.key) + yScale.bandwidth() / 2)
        .attr("dx", "0.3em")
        .attr("dy", "0.35em")
        .text(d => d.value.toFixed(2));
    bars.attr("fill", "steelblue");
    labels.attr("fill", "black")
          .attr("text-anchor", "start");
    // Perform any additional processing or manipulation here
  })
  .catch(error => {
    // Handle any errors that occur during the request
    console.error('Error:', error);
  });
