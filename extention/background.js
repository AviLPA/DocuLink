chrome.runtime.onInstalled.addListener(() => {
  console.log('Extension installed');
});

chrome.action.onClicked.addListener((tab) => {
  chrome.identity.getAuthToken({ interactive: true }, (token) => {
    if (token) {
      chrome.storage.local.set({ googleToken: token }, () => {
        chrome.tabs.create({ url: chrome.runtime.getURL('popup.html') });
      });
    }
  });
});

function loadGoogleApiScript(callback) {
  var script = document.createElement('script');
  script.src = 'https://apis.google.com/js/api.js';
  script.onload = callback;
  document.head.appendChild(script);
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'loadGooglePicker') {
    const token = message.token;

    loadGoogleApiScript(() => {
      gapi.load('picker', { 'callback': () => {
        var picker = new google.picker.PickerBuilder()
          .addView(google.picker.ViewId.DOCS_VIDEOS)
          .setOAuthToken(token)
          .setDeveloperKey('YOUR_DEVELOPER_KEY')
          .setCallback((data) => {
            if (data.action === google.picker.Action.PICKED) {
              var fileId = data.docs[0].id;
              sendResponse({ fileId });
            }
          })
          .build();
        picker.setVisible(true);
      }});
    });

    return true; // Indicates you wish to send a response asynchronously
  }
});
