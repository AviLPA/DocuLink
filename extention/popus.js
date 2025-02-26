document.getElementById('pick').addEventListener('click', () => {
    chrome.storage.local.get('googleToken', (result) => {
      const token = result.googleToken;
  
      chrome.runtime.sendMessage({ type: 'loadGooglePicker', token }, (response) => {
        if (chrome.runtime.lastError) {
          console.error(chrome.runtime.lastError.message);
          document.getElementById('output').innerHTML = 'Error loading Google Picker.';
          return;
        }
  
        const fileId = response.fileId;
        document.getElementById('output').innerHTML = 'Video ID: ' + fileId;
        processVideo(fileId, token);
      });
    });
  });
  
  function processVideo(fileId, token) {
    fetch(`https://www.googleapis.com/drive/v3/files/${fileId}?alt=media`, {
      headers: new Headers({ 'Authorization': 'Bearer ' + token })
    })
      .then(response => response.blob())
      .then(blob => {
        let formData = new FormData();
        formData.append('video', blob, 'video.mp4');
  
        fetch('http://localhost:5009/upload', {
          method: 'POST',
          body: formData
        })
          .then(response => response.json())
          .then(data => {
            document.getElementById('output').innerHTML = JSON.stringify(data);
          })
          .catch(error => {
            console.error('Error:', error);
            document.getElementById('output').innerHTML = 'Error processing video.';
          });
      })
      .catch(error => {
        console.error('Error fetching video:', error);
        document.getElementById('output').innerHTML = 'Error fetching video.';
      });
  }
  