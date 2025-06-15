document.addEventListener("DOMContentLoaded", function () {
  document
    .getElementById("detectButton")
    .addEventListener("click", detectPhishing);

  function detectPhishing() {
    showLoading(true);

    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      var url = tabs[0].url;

      updateWebsite(url);

      makePrediction(url)
        .then((isPhishing) => {
          updateButton(isPhishing);
          updateWebsiteDetails(isPhishing);
          setTimeout(() => {
            updateButton(isPhishing);
          }, 2000);
        })
        .catch((error) => {
          console.error("Error making prediction:", error);
        })
        .finally(() => {
          showLoading(false);
        });
    });
  }

  function updateButton(isPhishing) {
    var button = document.getElementById("detectButton");

    button.style.backgroundColor = "black";
    button.style.color = "white";

    if (isPhishing) {
      button.innerText = "Scam";
      button.style.backgroundColor = "red";
    } else {
      button.innerText = "Genuine!";
      button.style.backgroundColor = "green";
    }
  }
  function updateWebsite(url) {
    var websiteName = document.getElementById("website");

    websiteName.innerText = `Website: ${url}`;
  }

  function updateWebsiteDetails(isPhishing) {
    var websiteName = document.getElementById("website");
    var accuracyText = document.getElementById("accuracy-text");

    if (isPhishing) {
      websiteName.style.color = "red";
      accuracyText.style.color = "red";
    } else {
      websiteName.style.color = "green";
      accuracyText.style.color = "green";
    }
  }
  async function makePrediction(url) {
    try {
      const apiUrl = "http://127.0.0.1:5000/check-url";
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: url }),
      });
      // console.log(response);
      if (response.ok) {
        try {
          const data = await response.json();
          // console.log(data);
          if ("prediction" in data) {
            const predictionLikelihood = data["prediction"];
            updateAccuracyText(predictionLikelihood);
            return parseFloat(predictionLikelihood) >= 40;
          } else {
            console.error("Invalid response format:", data);
            return false;
          }
        } catch (error) {
          console.error("Invalid JSON in response:", error);
          return false;
        }
      } else {
        console.error("Request failed with status:", response.status);
        return false;
      }
    } catch (error) {
      console.error("Error making prediction:", error);
      return false;
    }
  }

  function updateAccuracyText(likelihood) {
    var accuracyText = document.getElementById("accuracy-text");
    accuracyText.innerText = `Likelihood: ${likelihood}`;
  }

  function showLoading(isLoading) {
    var loadingDiv = document.getElementById("loading");
    var detectButton = document.getElementById("detectButton");

    if (isLoading) {
      loadingDiv.style.display = "block";

      detectButton.disabled = true;
    } else {
      loadingDiv.style.display = "none";

      detectButton.disabled = false;
    }
  }
});
