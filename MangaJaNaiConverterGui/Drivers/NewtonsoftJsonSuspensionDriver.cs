using Newtonsoft.Json;
using ReactiveUI;
using ReactiveUI.Primitives;
using ReactiveUI.Primitives.Signals;
using System;
using System.IO;
using System.Text.Json.Serialization.Metadata;

namespace MangaJaNaiConverterGui.Drivers
{
    public class NewtonsoftJsonSuspensionDriver : ISuspensionDriver
    {
        private readonly string _file;
        public static readonly JsonSerializerSettings Settings = new()
        {
            TypeNameHandling = TypeNameHandling.All,
            Formatting = Formatting.Indented,
        };

        public NewtonsoftJsonSuspensionDriver(string file) => _file = file;

        public IObservable<RxVoid> InvalidateState()
        {
            if (File.Exists(_file))
                File.Delete(_file);
            return Signal.Return(RxVoid.Default);
        }

        public IObservable<object> LoadState()
        {
            var lines = File.ReadAllText(_file);
            var state = JsonConvert.DeserializeObject<object>(lines, Settings)!;
            return Signal.Return(state);
        }

        public IObservable<RxVoid> SaveState(object state)
        {
            var lines = JsonConvert.SerializeObject(state, Settings);
            File.WriteAllText(_file, lines);
            return Signal.Return(RxVoid.Default);
        }

        // ReactiveUI 23 added AOT/source-gen-friendly generic overloads to ISuspensionDriver (and 26
        // moved it off System.Reactive: RxVoid/Signal instead of Unit/Observable).
        // We keep using Newtonsoft (with TypeNameHandling.All) for polymorphic state, so the
        // JsonTypeInfo<T> metadata argument is ignored.
        public IObservable<RxVoid> SaveState<T>(T state) => SaveState((object)state!);

        public IObservable<RxVoid> SaveState<T>(T state, JsonTypeInfo<T> typeInfo) => SaveState((object)state!);

        public IObservable<T?> LoadState<T>(JsonTypeInfo<T> typeInfo)
        {
            var lines = File.ReadAllText(_file);
            var state = JsonConvert.DeserializeObject<T>(lines, Settings);
            return Signal.Return(state);
        }
    }
}
