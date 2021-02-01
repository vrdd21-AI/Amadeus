package com.example.amadeus;

import android.content.Context;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.text.TextUtils;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;

public class ChatActivity extends AppCompatActivity {
    private RecyclerView recyclerView;
    private RecyclerView.Adapter mAdapter;
    private RecyclerView.LayoutManager layoutManager;
    private ArrayList<ChatDTO> myDataset;
    private EditText editText;

    private Socket client = null;
    public InputStream inp;
    public OutputStream out;

    private String name = "";

    public boolean isStop = true; // for client read thread
    public boolean canAnswer = true;
    private String IP = NetworkConfigure.IP;
    //private String IP = "192.168.0.16";
    private int PORT = NetworkConfigure.PORT;

    public Context context;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chatting);

        context = this;

        Intent intent = getIntent();
        name = intent.getExtras().getString("name");

        editText = findViewById(R.id.edit_text);

        editText.setOnEditorActionListener(new EditText.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
                switch (actionId) {
                    case EditorInfo.IME_ACTION_DONE:
                        sendChat(editText);
                        break;
                    case EditorInfo.IME_ACTION_SEND:
                        sendChat(editText);
                        break;
                    default:
                        sendChat(editText);
                        // 기본 엔터키 동작
                        return false;
                }
                return true;
            }
        });

        myDataset = new ArrayList<ChatDTO>();
        recyclerView = (RecyclerView) findViewById(R.id.recycler_view);
        // use this setting to improve performance if you know that changes
        // in content do not change the layout size of the RecyclerView
        recyclerView.setHasFixedSize(true);

        // use a linear layout manager
        layoutManager = new LinearLayoutManager(this);
        recyclerView.setLayoutManager(layoutManager);

        if(name.equals("Daru")){
            myDataset.add(new ChatDTO(name, "What?"));
            editText.setText("Kurisu was stabbed ...");

        }

        if(name.equals("Mayushi")){
            myDataset.add(new ChatDTO(name, "Where is my Metal U-pa...?"));
            editText.setText("Metal Upa? don't find it.");
        }

        // specify an adapter (see also next example)
        mAdapter = new ChatAdapter(myDataset);
        recyclerView.setAdapter(mAdapter);

        if(name.equals("Kurisu"))
            connect(IP, PORT);
        else {
            editText.setEnabled(false);
        }
        Log.d("Test", name);
    }

    public void connect(final String ip, final int port){
        new AsyncTask<Void, String, Void>(){
            public String sConnectError = "";
            public String sClientError = "";
            @Override
            protected void onPostExecute(Void aVoid) {

                super.onPostExecute(aVoid);
            }

            @Override
            protected void onProgressUpdate(String... values) {
                super.onProgressUpdate(values);

                if(values == null)
                    return;
                Log.d("Test", values[0]);
                values[0] = values[0].substring(2, values[0].length());
                myDataset.add(new ChatDTO(name, values[0]));
                new Handler().postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        recyclerView.scrollToPosition(myDataset.size()-1);
                        mAdapter.notifyItemInserted(myDataset.size()-1);
                        canAnswer = true;
                    }
                }, 300);
            }

            @Override
            protected Void doInBackground(Void... voids) {
                try{
                    client = new Socket(ip, port);
                    inp = client.getInputStream();
                    out = client.getOutputStream();
                } catch (IOException e){
                    e.printStackTrace();
                    sConnectError = e.getMessage();
                    return null;
                }

                byte[] buffer = new byte[1024];
                int bytes = 0;

                try{
                    isStop = false;

                    while(isStop == false){
                        //String sMsg = clientIn.readUTF();
                        bytes = inp.read(buffer);
                        if(bytes < 0) // 여기 계속 -1 들어오던데 성능에 문제가 있을려난
                            break;
                        String sMsg = new String(buffer, 0, bytes);
                        if(TextUtils.isEmpty(sMsg) == false){
                            publishProgress(sMsg); // UI thread
                        }
                    }
                } catch(IOException e){
                    e.printStackTrace();
                    sClientError = e.getMessage();
                }
                return null;
            }
        }.execute();
    }

    public void sendMessage(final String sMsg){
        if(client == null){
            return;
        }

        new AsyncTask<Void, Integer, Void>(){
            private String sErr = "";
            @Override
            protected Void doInBackground(Void... voids) {
                try{
                    byte[] buffer = sMsg.getBytes();
                    out.write(buffer);
                } catch (IOException e){
                    e.printStackTrace();
                    sErr = e.getMessage();
                }
                return null;
            }

            @Override
            protected void onPostExecute(Void aVoid) {
                super.onPostExecute(aVoid);
            }
        }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
        // executeOnExecutor는 무슨 의미지
    }

    public void sendChat(View view){
        if(canAnswer == false || editText.getText().toString().length() == 0){
            return;
        }
        myDataset.add(new ChatDTO("Okabe", editText.getText().toString()));
        Log.d("Test", "Test: " + editText.getText().toString().length());

        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                recyclerView.scrollToPosition(myDataset.size()-1);
                mAdapter.notifyItemInserted(myDataset.size()-1);
            }
        }, 300);


        if(name.equals("Daru")) {
            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    myDataset.add(new ChatDTO(name, "it is real?"));

                }
            }, 1000);

            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    recyclerView.scrollToPosition(myDataset.size()-1);
                    mAdapter.notifyItemInserted(myDataset.size()-1);
                }
            }, 1300);

            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    myDataset.add(new ChatDTO(name, "operation start!"));

                }
            }, 2000);

            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    recyclerView.scrollToPosition(myDataset.size()-1);
                    mAdapter.notifyItemInserted(myDataset.size()-1);

                }
            }, 2300);



            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    Intent intent = new Intent(context, TestActivity.class);
                    startActivityForResult(intent, 1);

                }
            }, 3000);

        }

        canAnswer = false;
        sendMessage(editText.getText().toString());
        editText.setText("");

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try{
            if(client != null) {
                client.close();
                Log.d("Test", "closed");
            }
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void returnClicked(View view){
        Intent intent = new Intent();
        intent.putExtra("lasttext", myDataset.get(myDataset.size()-1).content);
        setResult(11, intent);
        finish();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode==1){
            Log.d("Test", "result code 10!");
            Intent intent = new Intent();
            setResult(10);
            finish();
        }
    }
}
