function useModel(modelName) {
    fetch(`/use_model?model=${modelName}`)
        .then(response => response.json())
        .then(data => {
            console.log(data.message);
        })
        .catch(error => {
            console.log(error);
        });
}
