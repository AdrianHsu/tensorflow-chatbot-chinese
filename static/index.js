(function(){
  // https://stackoverflow.com/questions/14521108/dynamically-load-js-inside-js

  var chat = {
    messageToSend: '',
    messageResponses: [
      'Why did the web developer leave the restaurant? Because of the table layout.',
      'How do you comfort a JavaScript bug? You console it.',
      'An SQL query enters a bar, approaches two tables and asks: "May I join you?"',
      'What is the most used language in programming? Profanity.',
      'What is the object-oriented way to become wealthy? Inheritance.',
      'An SEO expert walks into a bar, bars, pub, tavern, public house, Irish pub, drinks, beer, alcohol'
    ],
    init: function() {
      this.cacheDOM();
      this.bindEvents();
      this.render();
    },
    cacheDOM: function() {
      this.$chatHistory = $('.chat-history');
      this.$button = $('button');
      this.$textarea = $('#message-to-send');
      this.$chatHistoryList =  this.$chatHistory.find('ul');
    },
    bindEvents: function() {
      this.$button.on('click', this.addMessage.bind(this));
      this.$textarea.on('keyup', this.addMessageEnter.bind(this));
    },
    render: function() {
      this.scrollToBottom();
      if (this.messageToSend.trim() !== '') {
        var template = Handlebars.compile( $("#message-template").html());
        var context = { 
          messageOutput: this.messageToSend,
          time: this.getCurrentTime()
        };

        this.$chatHistoryList.append(template(context));
        this.scrollToBottom();
        this.$textarea.val('');
      }
    },
    renderResponse: function(msg) {
      this.scrollToBottom();
        
      var templateResponse = Handlebars.compile( $("#message-response-template").html());
      var contextResponse = { 
        response: msg,
        time: this.getCurrentTime()
      };
      this.$chatHistoryList.append(templateResponse(contextResponse));
      this.scrollToBottom();
    },
    addMessage: function() {
      this.messageToSend = this.$textarea.val();
      str = this.messageToSend;
      if(!str || /^\s*$/.test(str)) {
        this.$textarea.val(''); // make sure it's not empty string
        return;
      }
      console.log(this.messageToSend);
      socket.emit('message', this.messageToSend);
      this.render();         
    },
    addMessageEnter: function(event) {
        // enter was pressed
        if (event.keyCode === 13) {
          this.addMessage();
        }
    },
    scrollToBottom: function() {
       this.$chatHistory.scrollTop(this.$chatHistory[0].scrollHeight);
    },
    getCurrentTime: function() {
      return new Date().toLocaleTimeString().
              replace(/([\d]+:[\d]{2})(:[\d]{2})(.*)/, "$1$3");
    },
    getRandomItem: function(arr) {
      return arr[Math.floor(Math.random()*arr.length)];
    }
    
  };

  chat.init();
  var socket;  
  $.getScript("https://cdn.socket.io/socket.io-1.2.0.js", function(data, textStatus, jqxhr) {
    socket = io();
    socket.on('message', function(msg){
      console.log(msg); 
      chat.renderResponse(msg);     
    });
  });
  
})();