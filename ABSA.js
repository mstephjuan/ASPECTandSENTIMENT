fetch('http://127.0.0.1:5000/aspects')
  .then(response => response.json())
  .then(data => {
    // You can access the JSON data in the 'data' variable
    console.log(data);
     // Update the content of a <div> element with the JSON data
    const divElement = document.getElementById('dataContainer');
    divElement.textContent = JSON.stringify(data);
    // Perform any additional processing or manipulation here
  })
  .catch(error => {
    // Handle any errors that occur during the request
    console.error('Error:', error);
  });
